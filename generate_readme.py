from pathlib import Path

png_dir = Path("preview/png")
mp4_dir = Path("preview/mp4")
gif_dir = Path("preview/gif")
bvh_dirs = {
    "100%": Path("bvh/100_percent_scale"),
    "80%": Path("bvh/80_percent_scale")
}

width = 240  # width for image + video preview

png_files = sorted(png_dir.glob("*.png"))

rows = []
for png_file in png_files:
    base = png_file.stem
    mp4_file = mp4_dir / f"{base}.mp4"
    gif_file = gif_dir / f"{base}.gif"

    if not mp4_file.exists():
        continue

    # ---- Preview column (image + embedded video) ----
    preview_cell = (
        f'<div style="display:flex; gap:10px; align-items:center;">'
        f'<img src="{png_file}" width="{width}">'
        f'<img src="{gif_file}" width="{width}">'
        # github doesn't show videos...
        # f'<video width="{width}" controls preload="none" poster="{png_file}">'
        f'<source src="{mp4_file}" type="video/mp4">'
        f'</video>'
        f'</div>'
    )

    # ---- Files column (Video link + BVHs) ----
    links = []

    # Always include clickable video link (works on GitHub)
    links.append(f'[Video]({mp4_file})')

    for scale_name, dir_path in bvh_dirs.items():
        original = dir_path / f"{base}.bvh"
        mirrored = dir_path / f"{base}_mirrored.bvh"

        if original.exists():
            links.append(
                f'{scale_name} (original): '
                f'[{original.name}]({original})'
            )
        if mirrored.exists():
            links.append(
                f'{scale_name} (mirrored): '
                f'[{mirrored.name}]({mirrored})'
            )

    files_cell = "<br>".join(links)

    rows.append(f"| {preview_cell} | {files_cell} |")

# Write README.md
with open("README.md", "w") as f:
    f.write(
"""# Humanoid Omniskill Loco-manipulation Dataset (HOLD)
This project contains a collection of data collected at RAI using a Vivon optical motion capture system. 
The data has been collected with the intention of being useful for training a variety of robotic locomotion and manipulation tasks.
The term "Bone" used in this readme is synonymous with a "Joint" in the .bvh file format specification.

Mirrored data, as well as scaled data for smaller robots is provided for convenience. In the scaled data, all length units are at 80% scale, and time runs at 1/sqrt(0.8) the rate in order to preserve g=9.81m/s^2.
Pose data was not resampled in order to prevent needless quality loss - instead the frame time in the scaled .bvh has been modified, therefore it is important you consider 
this when reading from these files and appropriately interpolate poses if you wish to keep gravity at 9.81m/s^2 in the scaled data.

### Estimated Foot Bones
The original foot bones output by the motion capture software do not provide clear information about the ground plane interactions of the feet, and may be subject to errors due to miscalibration/marker movement. 
For this reason there are extra "Estimated" bones added to the files:

- LeftSoleHeel_Estimated  
- RightSoleHeel_Estimated  
- LeftSoleToe_Estimated  
- RightSoleToe_Estimated  

These bone positions & orientations are configured such that when the actor is standing with flat feet in double support: 

- The bones lie flat on the ground at a height around z=0.
- Z axis of the bone frame is gravity aligned.
- X axis of the bone frame points in toe pointing direction.

This is done by detecting double support using a velocity+height thresholding heurestic for the feet/toe bones 
and then estimating the constant foot bone pose which minimize various costs across all these states.
This process is not always perfect, as the motion capture volume may not have a correct floor calibration, and the heurestic itself can fail.
Some files have manually been specified to reuse the estimate from other skeletons from a similar take, because the heurestic has failed.

### Prop Bones
Some files have props which were motion captured. These are boxes and in some files, chairs. The extents of the props were not perfectly recorded and have been inferred. 
These props have bones parented to the "Z_UP_WORLD_ORIGIN" bone. The prop bones then have child bones which represent the 8 corners of the relevant bounding box via. their end site offset.
If you import these .bvh files into blender the extents of the boxes are clearly indicated by the tip of the bones. 
If you wish to deduce the scale of the box directly from the .bvh, 
you can read the end site offset for the <prop_name>_x_y_z_corner bone, as this will contain the half-lengths of the box. These are expressed in the frame of the prop bone (with name <prop_name>).
""")
    f.write("# Dataset Preview\n\n")
    f.write("This table shows a preview of BVH files in the dataset.\n\n")
    f.write("| Preview | Files |\n")
    f.write("|----------|-------|\n")
    f.write("\n".join(rows))
    f.write("\n# Generating this file\n\n")
    f.write("Run `python generate_readme.py` in the root project directory to generate this README.md from the corresponding files.\n")
