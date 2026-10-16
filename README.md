![HOLD: Humanoid Omniskill Loco-manipulation Dataset](imgs/hold-banner.gif)

HOLD is a dataset of human motion data collected at [RAI Institute](https://rai-inst.com/) in collaboration with [Boston Dynamics](https://bostondynamics.com/) using a [Vicon](https://www.vicon.com/) optical motion capture system. 

## What is included? 

In total, HOLD includes 4,053,854 frames across 552 human motion trajectories (276 originals and their mirrored counterparts), corresponding to 576.6 minutes of data (recorded at 120 Hz). The data includes: 
- Wide-ranging human locomotion on flat terrain, from small-step shuffling, standard walking, sidestepping, and turning to high-speed jogging and sprinting, across various headings and speeds to cover a wide range of body planar velocities.
- Whole-body human-object interactions, such as lifting, carrying, and pushing boxes, climbing boxes, and sitting on a chair.

The data is provided in BVH format under [`data/bvh`](data/bvh/). We post-processed the raw motion capture recordings to remove kinematic artifacts.

For convenience, we also provide 80%-scaled data that matches the dimensions of widely used commercial humanoid robots such as the Unitree G1. This reduces retargeting artifacts caused by the size mismatch between humans and these robots.
As an example of retargeting results, we provide motion data retargeted to the Unitree G1 robot under [`data/g1`](data/g1/).    

For a comprehensive preview of the dataset, see [docs/preview.md](docs/preview.md).

## How was the data collected and processed?

The whole-body human-object interaction recordings focused on diversity of behaviors, while the flat-terrain locomotion recordings focused on covering a wide range of planar body velocities, with transitions between speeds and gait modalities. For locomotion in particular, we collected data iteratively: whenever the locomotion policies we developed with HOLD showed capability gaps caused by data sparsity, we collected additional examples. For instance, if sidestepping performance was suboptimal, we recorded more sidestepping data.

Every BVH file in HOLD was post-processed to remove kinematic artifacts. Because of sensor noise and tracking errors that occasionally occur in optical motion capture systems, foot bones can exhibit slight misalignments, which makes the timing of ground-contact events ambiguous. To address this, we identify periods when the actor is stationary. During those periods, we estimate the ground plane and the corresponding sole positions from the kinematic configuration and the foot height relative to the coordinate origin.

See [docs/process.md](docs/process.md) to learn more about the post-processing. For more details on how we collect, process, and use the HOLD dataset, see the paper [HuMBLE: Human Motion-Driven Behavior Learning for Embodied Locomotion](https://arxiv.org/abs/2610.10489).

## Getting started 

We distribute this data using [`git-lfs`](https://git-lfs.com/). To install Git LFS (Debian/Ubuntu):
```bash
sudo apt-get install git-lfs
git lfs install
```
Install Git LFS before cloning so the large files are downloaded. If you already cloned the repository, run `git lfs pull` to fetch them. For other platforms, see the [git-lfs installation guide](https://git-lfs.com/).

Then download the dataset by cloning this repository:
```bash
git clone https://github.com/rai-opensource/hold.git
```

We recommend using [Blender](https://www.blender.org/) to visually inspect the BVH animations. To visualize the G1-retargeted trajectories, use the simple [MuJoCo](https://mujoco.org/)-based viewer described in [docs/g1.md](docs/g1.md).

## Research using HOLD

HOLD was used in the joint RAI Institute–Boston Dynamics work [HuMBLE: Human Motion-Driven Behavior Learning for Embodied Locomotion](https://arxiv.org/abs/2610.10489), which develops fully steerable human-like locomotion for humanoid robots. 
Notably, this locomotion was demonstrated live on stage at the [launch event for Boston Dynamics' new Atlas robot at CES 2026](https://youtu.be/YA_TlJv0Uro?si=36n7i9BSnQpWSiKR&t=22). 

We release HOLD as open data to foster broad progress in humanoid research under the [Creative Commons Attribution 4.0 International license](LICENSE).

If you use this work in your research or project, please consider citing it using the following BibTeX entry:

```
@misc{raibd2026hold,
  author = {{RAI Institute} and {Boston Dynamics}},
  title = {HOLD: Humanoid Omniskill Loco-manipulation Dataset},
  month = {October},
  year = {2026},
  url = {https://github.com/rai-opensource/hold}
}
```

## Acknowledgments

HOLD is the result of a collaboration between the RAI Institute and Boston Dynamics. We thank all the contributors who were involved in the collection, processing, and release of this data. We especially thank Jonathan Foster for serving as our human performer.

![RAI Institute and Boston Dynamics logos](imgs/bd-rai-logos.png)

If you have any questions about the dataset, please contact [Dongho Kang](mailto:dkang@rai-inst.com).

