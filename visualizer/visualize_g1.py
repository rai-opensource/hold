#!/usr/bin/env python3
"""Play HOLD G1 trajectories (data/g1/**/*.h5) in a MuJoCo window with a time slider.

Usage:
    uv run --project visualizer visualizer/visualize_g1.py data/g1/locomotion/walk
    uv run --project visualizer visualizer/visualize_g1.py data/g1/locomotion/sprint/s0_sprint_001.h5

Drag the slider at the bottom of the window to scrub through the motion, and click a file in the
list on the right to play it.
Keys: SPACE pause/resume, LEFT/RIGHT seek -/+1 s, N/P next/previous motion,
      UP/DOWN faster/slower, R restart, ESC quit.
Mouse: drag the scene to rotate the camera, scroll to zoom (or to scroll the file list).
"""

import argparse
import os
import re
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path

import glfw
import h5py
import mujoco
import numpy as np

# Unitree G1 (29 DoF) model from MuJoCo Menagerie, downloaded once and cached.
MENAGERIE_COMMIT = "62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35"
MENAGERIE_URL = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/"
    f"{MENAGERIE_COMMIT}/unitree_g1"
)
DEFAULT_CACHE = Path.home() / ".cache" / "hold" / f"unitree_g1_{MENAGERIE_COMMIT[:8]}"

# Slider layout in framebuffer pixels.
BAR_HEIGHT = 56
LABEL_WIDTH = 220
BAR_PAD = 28

# File list panel on the right, in framebuffer pixels.
PANEL_WIDTH = 420
TAB_WIDTH = 170  # the "Files" tab shown while the list is hidden
ROW_HEIGHT = 26
HEADER_HEIGHT = 32
SCROLLBAR_WIDTH = 8
CHAR_WIDTH = 10.0  # approximate width of a character in the viewer's font


def fetch_model(cache_dir: Path) -> Path:
    """Returns the path to scene.xml, downloading the G1 model if needed."""
    scene = cache_dir / "scene.xml"
    if scene.exists():
        return scene

    print(f"Downloading the Unitree G1 model to {cache_dir} ...")
    tmp = Path(tempfile.mkdtemp(prefix="g1_model_"))

    def download(name: str) -> None:
        dst = tmp / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{MENAGERIE_URL}/{name}", dst)

    try:
        for name in ("g1.xml", "scene.xml", "LICENSE"):
            download(name)
        for mesh in sorted(set(re.findall(r'<mesh\b[^>]*\bfile="([^"]+)"', (tmp / "g1.xml").read_text()))):
            download(f"assets/{mesh}")
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.move(str(tmp), cache_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return scene


class Motion:
    """A HOLD G1 trajectory converted to MuJoCo qpos (free joint on the pelvis)."""

    def __init__(self, path: Path, model: mujoco.MjModel):
        self.path = path
        with h5py.File(path, "r") as f:
            traj = f["robot_data/trajectory_data"]
            self.time = traj["time"][:]
            joint_pos = traj["joints/pos"][:]
            joint_names = [n.decode() for n in traj["joints/labels"][:]]
            body_names = [n.decode() for n in traj["bodies/labels"][:]]
            base = body_names.index("base")
            base_pos = traj["bodies/pos_w"][:, base]
            base_quat = traj["bodies/quat_w"][:, base]  # [w, x, y, z]

        # Joint order differs between files, so map by name.
        qadr = [model.joint(n).qposadr[0] for n in joint_names]
        self.qpos = self._to_qpos(model, qadr, joint_pos, base_pos, base_quat)

    @staticmethod
    def _to_qpos(model, qadr, joint_pos, base_pos, base_quat) -> np.ndarray:
        # The "base" frame in the data coincides with the G1 torso_link, whereas the
        # MJCF root is the pelvis. Recover the pelvis pose from the torso pose using
        # forward kinematics of the waist joints.
        data = mujoco.MjData(model)
        torso = model.body("torso_link").id
        qpos = np.zeros((len(joint_pos), model.nq))
        torso_quat_inv = np.zeros(4)
        pelvis_quat = np.zeros(4)
        offset = np.zeros(3)
        for i in range(len(joint_pos)):
            data.qpos[:] = 0.0
            data.qpos[3] = 1.0
            data.qpos[qadr] = joint_pos[i]
            mujoco.mj_kinematics(model, data)
            mujoco.mju_negQuat(torso_quat_inv, data.xquat[torso])
            mujoco.mju_mulQuat(pelvis_quat, base_quat[i], torso_quat_inv)
            mujoco.mju_rotVecQuat(offset, data.xpos[torso], pelvis_quat)
            qpos[i, qadr] = joint_pos[i]
            qpos[i, :3] = base_pos[i] - offset
            qpos[i, 3:7] = pelvis_quat
        return qpos

    @property
    def duration(self) -> float:
        return float(self.time[-1])

    def sample(self, t: float) -> np.ndarray:
        """Returns qpos at time t, interpolated between the two surrounding frames."""
        i = min(int(np.searchsorted(self.time, t, side="right")) - 1, len(self.time) - 2)
        i = max(i, 0)
        a = float(np.clip((t - self.time[i]) / (self.time[i + 1] - self.time[i]), 0.0, 1.0))
        q0, q1 = self.qpos[i], self.qpos[i + 1]
        qpos = (1.0 - a) * q0 + a * q1
        if np.dot(q0[3:7], q1[3:7]) < 0.0:  # take the short way around for the root quaternion
            qpos[3:7] = (1.0 - a) * q0[3:7] - a * q1[3:7]
        qpos[3:7] /= np.linalg.norm(qpos[3:7])
        return qpos


def collect_files(paths: list[Path]) -> list[Path]:
    files = []
    for p in paths:
        if p.is_dir():
            files += sorted(p.rglob("*.h5"))
        elif p.suffix == ".h5":
            files.append(p)
        else:
            raise SystemExit(f"Not an .h5 file or directory: {p}")
    if not files:
        raise SystemExit("No .h5 files found.")
    return files


def file_labels(files: list[Path]) -> list[str]:
    """Names for the file list: paths relative to the directory the files have in common."""
    resolved = [f.resolve() for f in files]
    root = Path(os.path.commonpath([f.parent for f in resolved]))
    return [f.relative_to(root).as_posix() for f in resolved]


class Player:
    """Playback state: which motion is loaded and where we are in it."""

    def __init__(self, model: mujoco.MjModel, files: list[Path], speed: float):
        self.model = model
        self.data = mujoco.MjData(model)
        self.files = files
        self.labels = file_labels(files)
        self.speed = speed
        self.index = 0
        self.paused = False
        self.scrubbing = False  # true while the user drags the slider
        self.clock = 0.0  # motion time in seconds
        self.requested = None  # index of a motion to switch to, applied by load_pending()
        self.motion = Motion(files[0], model)
        self._announce()
        self._last = time.perf_counter()

    def _announce(self) -> None:
        print(f"[{self.index + 1}/{len(self.files)}] {self.motion.path}")

    def seek(self, t: float) -> None:
        self.clock = float(np.clip(t, 0.0, self.motion.duration))

    def request(self, index: int) -> None:
        self.requested = index % len(self.files)

    def request_relative(self, steps: int) -> None:
        self.request((self.index if self.requested is None else self.requested) + steps)

    def load_pending(self) -> None:
        """Loads the requested motion. Loading takes a moment, so the window is not updated meanwhile."""
        index, self.requested = self.requested, None
        if index is not None and index != self.index:
            self.index = index
            self.motion = Motion(self.files[index], self.model)
            self._announce()
            self.clock = 0.0
        self._last = time.perf_counter()  # don't count the loading time as playback time

    def step(self) -> None:
        now = time.perf_counter()
        if not (self.paused or self.scrubbing):
            # Motion time follows the timestamps stored in the H5 file (real time at 1x).
            # Cap dt so a stall (e.g. window startup) doesn't skip ahead in the motion.
            self.clock += min(now - self._last, 0.1) * self.speed
            self.clock %= self.motion.duration
        self._last = now
        self.data.qpos[:] = self.motion.sample(self.clock)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)  # light poses are needed for lighting and shadows


class Viewer:
    """A minimal GLFW window that renders the model and a horizontal time slider at the bottom."""

    def __init__(self, model: mujoco.MjModel, player: Player, size=(1280, 800), visible=True):
        if not glfw.init():
            raise SystemExit("Failed to initialize GLFW.")
        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.VISIBLE, visible)
        self.window = glfw.create_window(size[0], size[1], "HOLD G1 visualizer", None, None)
        if not self.window:
            glfw.terminate()
            raise SystemExit("Failed to create a GLFW window.")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.model = model
        self.player = player
        self.opt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self.cam)
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 3.0, -15.0, 140.0
        self.scn = mujoco.MjvScene(model, maxgeom=2000)
        self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

        self._drag = None  # "slider", "camera" or None
        self._cursor = (0.0, 0.0)  # window coordinates
        self._scroll = 0.0  # first visible row of the file list
        self._panel_open = True
        self._last_index = player.index
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)

    # Input handling.

    def _on_key(self, window, key, scancode, action, mods) -> None:
        if action == glfw.RELEASE:
            return
        player = self.player
        if key == glfw.KEY_SPACE and action == glfw.PRESS:
            player.paused = not player.paused
        elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
            player.seek(player.clock + (1.0 if key == glfw.KEY_RIGHT else -1.0))
        elif key in (glfw.KEY_N, glfw.KEY_P) and action == glfw.PRESS:
            player.request_relative(1 if key == glfw.KEY_N else -1)
        elif key == glfw.KEY_UP:
            player.speed = min(player.speed * 1.25, 8.0)
        elif key == glfw.KEY_DOWN:
            player.speed = max(player.speed / 1.25, 0.05)
        elif key == glfw.KEY_R and action == glfw.PRESS:
            player.seek(0.0)
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

    def _framebuffer_pos(self, x: float, y: float) -> tuple[float, float]:
        """Converts window coordinates (origin top-left) to framebuffer pixels (origin bottom-left)."""
        win_w, win_h = glfw.get_window_size(self.window)
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        return x * fb_w / max(win_w, 1), fb_h - y * fb_h / max(win_h, 1)

    def _track(self, fb_width: int) -> tuple[float, float]:
        return LABEL_WIDTH, max(fb_width - BAR_PAD, LABEL_WIDTH + 1)

    def _seek_to_x(self, fb_x: float) -> None:
        x0, x1 = self._track(glfw.get_framebuffer_size(self.window)[0])
        self.player.seek(np.clip((fb_x - x0) / (x1 - x0), 0.0, 1.0) * self.player.motion.duration)

    # File list geometry. Rows are counted from the top of the list.

    def _panel_width(self, fb_w: int) -> int:
        """Width of the open file list, or 0 while it is hidden."""
        return min(PANEL_WIDTH, fb_w // 2) if self._panel_open else 0

    def _header_rect(self, fb_w: int, fb_h: int) -> tuple[int, int, int, int]:
        """The "Files" header (x, y, width, height). While the list is hidden it is a small tab."""
        width = self._panel_width(fb_w) or TAB_WIDTH
        return fb_w - width, fb_h - HEADER_HEIGHT, width, HEADER_HEIGHT

    def _in_header(self, fb_x: float, fb_y: float) -> bool:
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        x, y, w, h = self._header_rect(fb_w, fb_h)
        return x <= fb_x <= x + w and y <= fb_y <= y + h

    def _visible_rows(self, fb_h: int) -> int:
        return max((fb_h - BAR_HEIGHT - HEADER_HEIGHT) // ROW_HEIGHT, 1)

    def _clamp_scroll(self, fb_h: int) -> None:
        max_scroll = max(len(self.player.files) - self._visible_rows(fb_h), 0)
        self._scroll = float(np.clip(self._scroll, 0, max_scroll))

    def _in_panel(self, fb_x: float, fb_y: float) -> bool:
        fb_w = glfw.get_framebuffer_size(self.window)[0]
        return fb_x >= fb_w - self._panel_width(fb_w) and fb_y >= BAR_HEIGHT

    def _row_at(self, fb_x: float, fb_y: float) -> int | None:
        """Index of the file under the given framebuffer position, if any."""
        fb_h = glfw.get_framebuffer_size(self.window)[1]
        if not self._in_panel(fb_x, fb_y) or fb_y > fb_h - HEADER_HEIGHT:
            return None
        row = int((fb_h - HEADER_HEIGHT - fb_y) // ROW_HEIGHT) + int(self._scroll)
        return row if row < len(self.player.files) else None

    def _on_mouse_button(self, window, button, action, mods) -> None:
        if button != glfw.MOUSE_BUTTON_LEFT:
            return
        if action == glfw.RELEASE:
            self._drag = None
            self.player.scrubbing = False
            return
        self._cursor = glfw.get_cursor_pos(window)
        fb_x, fb_y = self._framebuffer_pos(*self._cursor)
        if fb_y < BAR_HEIGHT:
            self._drag = "slider"
            self.player.scrubbing = True
            self._seek_to_x(fb_x)
        elif self._in_header(fb_x, fb_y):
            self._panel_open = not self._panel_open
        elif self._in_panel(fb_x, fb_y):
            row = self._row_at(fb_x, fb_y)
            if row is not None:
                self.player.request(row)
        else:
            self._drag = "camera"

    def _on_cursor(self, window, x, y) -> None:
        last_x, last_y = self._cursor
        self._cursor = (x, y)
        if self._drag == "slider":
            self._seek_to_x(self._framebuffer_pos(x, y)[0])
        elif self._drag == "camera":
            win_h = max(glfw.get_window_size(window)[1], 1)
            self.cam.azimuth -= 180.0 * (x - last_x) / win_h
            self.cam.elevation = float(np.clip(self.cam.elevation - 180.0 * (y - last_y) / win_h, -89.0, 89.0))

    def _on_scroll(self, window, dx, dy) -> None:
        if self._in_panel(*self._framebuffer_pos(*self._cursor)):
            self._scroll -= 3.0 * dy
            self._clamp_scroll(glfw.get_framebuffer_size(window)[1])
        else:
            self.cam.distance = float(np.clip(self.cam.distance * 0.9**dy, 0.5, 20.0))

    # Drawing.

    def draw(self, status: str | None = None) -> None:
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        if fb_w <= 0 or fb_h <= BAR_HEIGHT:
            return
        player = self.player
        self.cam.lookat[:] = player.data.qpos[:3]  # follow the pelvis; rotate/zoom stay free

        scene_rect = mujoco.MjrRect(0, BAR_HEIGHT, fb_w - self._panel_width(fb_w), fb_h - BAR_HEIGHT)
        mujoco.mjv_updateScene(
            self.model, player.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL.value, self.scn
        )
        mujoco.mjr_render(scene_rect, self.scn, self.ctx)

        if status is None:
            status = "paused" if player.paused else "playing"
        info = f"[{player.index + 1}/{len(player.files)}] {player.motion.path.name}\n{player.speed:.2f}x\n{status}"
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, scene_rect,
            "File\nSpeed\nStatus", info, self.ctx,
        )
        help_text = (
            "Space: pause\nLeft/Right: seek 1 s\nN/P: next/prev motion\nUp/Down: speed\nR: restart\n"
            "Click a file: play it\nClick Files header: hide list"
        )
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, scene_rect, help_text, "", self.ctx
        )
        self._draw_panel(fb_w, fb_h)
        self._draw_slider(fb_w, fb_h)

    def _draw_text_box(self, x, y, w, h, text, color, text_color=(1.0, 1.0, 1.0)) -> None:
        mujoco.mjr_rectangle(mujoco.MjrRect(int(x), int(y), int(w), int(h)), *color)
        # mjr_text positions are relative to the viewport of the previous draw call, which is
        # the rectangle just drawn.
        mujoco.mjr_text(mujoco.mjtFont.mjFONT_NORMAL, text, self.ctx, 10 / w, 0.0, *text_color)

    def _draw_panel(self, fb_w: int, fb_h: int) -> None:
        player = self.player
        n = len(player.files)
        width = self._panel_width(fb_w)
        x = fb_w - width
        rows = self._visible_rows(fb_h)

        hovered = self._in_header(*self._framebuffer_pos(*self._cursor))
        header_color = (0.2, 0.2, 0.2, 1.0) if hovered else (0.05, 0.05, 0.05, 1.0)
        if not self._panel_open:
            self._draw_text_box(*self._header_rect(fb_w, fb_h), f"< Files ({n})", header_color)
            return

        if player.index != self._last_index:  # keep the playing file in view when it changes
            self._last_index = player.index
            if not self._scroll <= player.index < self._scroll + rows:
                self._scroll = player.index - rows // 2
        self._clamp_scroll(fb_h)
        first = int(self._scroll)

        mujoco.mjr_rectangle(mujoco.MjrRect(x, BAR_HEIGHT, width, fb_h - BAR_HEIGHT), 0.1, 0.1, 0.1, 1.0)
        self._draw_text_box(*self._header_rect(fb_w, fb_h), f"> Files ({n})", header_color)

        hover = self._row_at(*self._framebuffer_pos(*self._cursor))
        text_width = width - SCROLLBAR_WIDTH
        max_chars = int((text_width - 20) / CHAR_WIDTH)
        for k in range(min(rows, n - first)):
            i = first + k
            label = player.labels[i]
            if len(label) > max_chars:  # keep the end of the path, which tells files apart
                label = "..." + label[len(label) - (max_chars - 3):]
            if i == player.index:
                color = (0.2, 0.45, 0.8, 1.0)
            elif i == hover:
                color = (0.25, 0.25, 0.25, 1.0)
            else:
                color = (0.13, 0.13, 0.13, 1.0) if i % 2 else (0.1, 0.1, 0.1, 1.0)
            y = fb_h - HEADER_HEIGHT - (k + 1) * ROW_HEIGHT
            self._draw_text_box(x, y, text_width, ROW_HEIGHT, label, color)

        if n > rows:
            track = rows * ROW_HEIGHT
            thumb = max(track * rows / n, 20)
            top = fb_h - HEADER_HEIGHT - (track - thumb) * first / (n - rows)
            mujoco.mjr_rectangle(
                mujoco.MjrRect(fb_w - SCROLLBAR_WIDTH, int(top - thumb), SCROLLBAR_WIDTH, int(thumb)), 0.5, 0.5, 0.5, 1.0
            )

    def _draw_slider(self, fb_w: int, fb_h: int) -> None:
        player = self.player
        x0, x1 = self._track(fb_w)
        cy = BAR_HEIGHT / 2
        knob_x = x0 + (x1 - x0) * player.clock / player.motion.duration

        def rect(x, y, w, h, color):
            mujoco.mjr_rectangle(mujoco.MjrRect(int(x), int(y), max(int(w), 1), int(h)), *color)

        rect(0, 0, fb_w, BAR_HEIGHT, (0.12, 0.12, 0.12, 1.0))
        rect(x0, cy - 3, x1 - x0, 6, (0.35, 0.35, 0.35, 1.0))
        rect(x0, cy - 3, knob_x - x0, 6, (0.25, 0.6, 1.0, 1.0))
        rect(knob_x - 6, cy - 12, 12, 24, (0.95, 0.95, 0.95, 1.0))
        label = f"{player.clock:7.2f} / {player.motion.duration:.2f} s"
        # mjr_text positions are relative to the viewport of the previous draw call, so draw
        # the label background first to make that viewport the label area.
        rect(0, 0, LABEL_WIDTH, BAR_HEIGHT, (0.12, 0.12, 0.12, 1.0))
        mujoco.mjr_text(
            mujoco.mjtFont.mjFONT_NORMAL, label, self.ctx, 20 / LABEL_WIDTH, (cy - 7) / BAR_HEIGHT, 1.0, 1.0, 1.0
        )

    def run(self) -> None:
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            if self.player.requested is not None:
                self.draw(status="loading...")
                glfw.swap_buffers(self.window)
                self.player.load_pending()
            self.player.step()
            self.draw()
            glfw.swap_buffers(self.window)

    def close(self) -> None:
        self.ctx.free()
        glfw.destroy_window(self.window)
        glfw.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="H5 files and/or directories to search for .h5 files")
    parser.add_argument("--speed", type=float, default=1.0, help="initial playback speed (default: 1.0)")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_CACHE,
                        help="directory with the G1 MJCF (scene.xml); downloaded here if missing")
    args = parser.parse_args()

    files = collect_files(args.paths)
    model = mujoco.MjModel.from_xml_path(str(fetch_model(args.model_dir)))
    viewer = Viewer(model, Player(model, files, args.speed))
    try:
        viewer.run()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
