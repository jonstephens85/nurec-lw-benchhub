# nurec-lw-benchhub

> Use NVIDIA NuRec captures as photoreal environments for [LW-BenchHub](https://github.com/LightwheelAI/LW-BenchHub) robotics tasks.

<!-- HERO IMAGE: replace with high-res screenshot of LeRobot on patio table -->
![LeRobot SO100 mounted in a NuRec scene, lifting a cube in a Gaussian splat environment](images/hero_lerobot_patio.png)

This guide shows how to use a NuRec scene as a photorealistic simulation environment inside LW-BenchHub.

The example uses the SO-101 LeRobot arm in the open-source garden scene to get started with a simple object-lifting task. The same pattern can be adapted to your own captured scenes, different robots, different object placements, and other LW-BenchHub tasks.

The goal is to give you the shortest working path from a NuRec scene to a running robotics training or evaluation task in LW-BenchHub.

---

## Workflow Overview

1. Capture a real-world scene on your phone (~50-200 photos)
2. Run COLMAP for camera poses
3. Train 3DGUT to produce a USDZ
4. Open the USDZ in Isaac Sim, add a ground plane + table collider, save
5. Drop four small files into your LW-BenchHub install (one task class, one CSV row, two YAMLs)
6. Add a single decorator line to enable RL on the new task
7. Run training or eval with `--task_config lerobot_liftobj_freeform_state[_play]`

---

## How the integration works

A NuRec scene drops into LW-BenchHub via two mechanisms, both of which already exist in the codebase but aren't documented for this use case:

**1. The `local_scene_path` hook.** When LW-BenchHub's `LwScene._setup_config` sees a `.usd` scene name (rather than a robocasa string like `robocasakitchen-9-8`), it sets `local_scene_path` and bypasses Lightwheel's cloud-hosted floorplan loader. The USD gets loaded directly. You activate this by passing your USD's filesystem path as the `layout` field in your task YAML.

**2. The CSV-based pose system.** Robot spawn pose is fetched at runtime via `csv_loader.load_robot_pose(robot, layout, task)`. The CSV is keyed by string equality on those three fields. A new CSV file dropped under `configs/` is auto-discovered. Cube spawn position is handled separately via `fix_object_pose_cfg` — an absolute-world-coords override on the task's placement.

The complication: most LW-BenchHub tasks (including `LiftObj`) anchor object placement and robot reference poses to robocasa fixtures (counter, cabinet, sink). A NuRec scene has none of those. The fix is a small subclass — `LiftObjFreeform` — that overrides the fixture-lookup methods and uses absolute world coordinates instead.

```
┌─────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│  Phone capture  │ ──▶ │  COLMAP + 3DGUT   │ ──▶ │   USDZ + USD     │
└─────────────────┘     └───────────────────┘     │  (splat +        │
                                                  │   colliders)     │
                                                  └──────────────────┘
                                                            │
                                                            ▼
┌─────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│  Training or    │ ◀── │   LW-BenchHub     │ ◀── │  LiftObjFreeform │
│  eval rollout   │     │   (Isaac Lab)     │     │  + CSV pose      │
└─────────────────┘     └───────────────────┘     │  + task YAML     │
                                                  └──────────────────┘
```

---

## Prerequisites

**Hardware**
- Linux (Ubuntu 22.04 tested; 24.04 works with a GCC-11 workaround)
- NVIDIA RTX GPU.
- ~50GB free disk for splat training outputs and LW-BenchHub assets

**Software**
- [COLMAP](https://colmap.github.io/install.html)
- [3DGRUT](https://github.com/nv-tlabs/3dgrut) — recommended install via UV
- [LW-BenchHub](https://github.com/LightwheelAI/LW-BenchHub)
- Isaac Sim 5.0+

You'll keep the 3DGRUT environment and the LW-BenchHub environment isolated. They use different CUDA versions, and 3DGRUT only runs during the scene-prep phase. Once you have a USDZ, you live entirely in the LW-BenchHub environment.

---

## Part 1 — Prepare Data for NuRec / 3DGRUT Scene Reconstruction

> **Canonical install instructions** for 3DGRUT and 3DGUT live in the [`nv-tlabs/3dgrut`](https://github.com/nv-tlabs/3dgrut) repo. The summary here is enough to get a USDZ; see upstream for the most current setup, especially around CUDA versioning and Blackwell GPU support.

For this example, I use the `garden` scene from the MipNeRF360 dataset as the starting point. That keeps the tutorial focused on the LW-BenchHub integration instead of phone capture and reconstruction quality.

You can also use your own captured scene. The workflow is the same once you have a trained 3DGRUT / NuRec export: open the USDZ in Isaac Sim, add physics proxies, save a USD, and point LW-BenchHub at that scene.

If you already have a NuRec or 3DGRUT scene exported as USDZ, you can skip ahead to Part 2.

### 1.1 Get the garden dataset

Download the MipNeRF360 `garden` scene from Hugging Face:

https://huggingface.co/datasets/mileleap/mipnerf360

Place it somewhere stable, for example:

```bash
mkdir -p ~/datasets/mipnerf360
# download / extract garden into:
# ~/datasets/mipnerf360/garden
```

### Optional: process your own dataset

If you want to use your own captured scene instead of the MipNeRF360 garden scene, you first need to process your images with COLMAP.

#### **Capture**

Smartphone works well for initial testing. Aim for ~60% overlap between adjacent images, slow loop around your subject with multiple heights, lock focus and exposure if your phone allows it. Convert HEIC → JPG before COLMAP if you're on iPhone.

For tabletop manipulation scenes, capture both the wider room context *and* close-ups of the surface where the robot will operate. The denser feature coverage on the manipulation surface pays off in splat sharpness exactly where you need it.

#### **Processing Data**

To install COLMAP:

```bash
sudo apt-get update
sudo apt-get install -y colmap
```

Verify: `colmap --help | head -1` should print COLMAP's banner.

The easiest way to start is with COLMAP's Automatic Reconstruction feature:

1. Launch COLMAP
2. Select **Reconstruction → Automatic Reconstruction**
3. Select your workspace folder and images folder
4. Important: select either the `PINHOLE` or `SIMPLE_PINHOLE` camera model for 3DGUT compatibility
5. Start the reconstruction

Once COLMAP finishes, verify that the sparse reconstruction looks coherent before moving on to 3DGRUT / 3DGUT training.

#### **Using COLMAP from the command line**

For more control or automation, you can also run COLMAP from the command line.

```bash
mkdir -p ./colmap/sparse

# Feature detection and extraction
colmap feature_extractor \
    --database_path ./colmap/database.db \
    --image_path ./images/ \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model PINHOLE \
    --SiftExtraction.max_image_size 2000 \
    --SiftExtraction.estimate_affine_shape 1 \
    --SiftExtraction.domain_size_pooling 1

# Feature matching
colmap exhaustive_matcher \
    --database_path ./colmap/database.db \
    --SiftMatching.use_gpu 1

# Sparse reconstruction
colmap mapper \
    --database_path ./colmap/database.db \
    --image_path ./images/ \
    --output_path ./colmap/sparse

# Visualize for verification
colmap gui \
    --import_path ./colmap/sparse/0 \
    --database_path ./colmap/database.db \
    --image_path ./images/
```

#### **Command parameters**

- `database_path`: path to the COLMAP database file
- `image_path`: directory containing your photos
- `ImageReader.single_camera`: assumes all images come from the same camera
- `ImageReader.camera_model`: camera model used for reconstruction. Use `PINHOLE` or `SIMPLE_PINHOLE` for 3DGUT compatibility
- `SiftExtraction.max_image_size`: maximum image dimension used during feature extraction
- `output_path`: directory where COLMAP writes the sparse reconstruction

#### **COLMAP output**

Once complete, you should have:

- A sparse point cloud of the scene
- Camera pose data for the registered images
- A project folder containing:
  - `database.db`: COLMAP database
  - `images/`: original photos
  - `sparse/`: reconstruction data

For 3DGUT training, the important thing is that your scene folder contains the image data and COLMAP sparse reconstruction in a structure that 3DGRUT can read.

### 1.2 Install 3DGRUT (UV path)

UV install is recommended — it's significantly faster than conda and handles dependencies more cleanly.

```bash
# Install UV if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone 3DGRUT
git clone --recursive https://github.com/nv-tlabs/3dgrut.git
cd 3dgrut

# Install OpenGL headers (required for the playground viewer)
sudo apt-get install libgl1-mesa-dev

# Run the UV install script (system CUDA path)
./install_env_uv.sh
source .venv/bin/activate
```

For other install paths (conda-managed CUDA, local-venv CUDA, Blackwell GPUs), see the [3DGRUT README](https://github.com/nv-tlabs/3dgrut). On Ubuntu 24.04 with system GCC 13, you may need to `sudo apt-get install gcc-11 g++-11` and use a CUDA-12.8.1 install path — the upstream README has the current recipe.

Verify the install:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.version.cuda)"
python -c "import threedgrut; print('3DGRUT OK')"
```

### 1.3 Train 3DGUT and export USDZ

The MCMC densification config (`apps/colmap_3dgut_mcmc.yaml`) gives sharper thin structures and is what the NuRec mono workflow recommends.

```bash
cd /path/to/3dgrut 
python train.py \
    --config-name apps/colmap_3dgut_mcmc.yaml \
    path=/path/to/scene/colmap \
    out_dir=/path/to/scene/output \
    experiment_name=my_scene_v1 \
    export_ply.enabled=true \
    export_usd.enabled=true \
    export_usd.apply_normalizing_transform=true
```

Three flags matter for the LW-BenchHub handoff:

- `export_ply.enabled=true` produces `export_last.ply` a 3DGS native file output. De
- `export_usdz.enabled=true` produces `convert.usdz` (or sometimes `export_last.usdz` depending on version) — the file Isaac Sim will load
- `export_usdz.apply_normalizing_transform=true` centers and roughly scales the scene near the origin
- **Note:** `apply_normalizing_transform` does *not* guarantee the floor sits at z=0 or that the scale matches real-world meters. You will almost certainly need to re-anchor and possibly rescale in Isaac Sim. Plan for it.

**IMPORTANT**

The expored usdz file may not be compatible with Isaac Sim. If the file does not render in Isaac Sim, convert the PLY file to a compatible USD format using:

`python -m threedgrut.export.scripts.ply_to_usd path/to/your/model.ply --output_file path/to/output.usdz`

Output paths to remember after training completes:

- `convert.usdz` — the splat data, referenced from your scene USD
- `ckpt_last.pt` — the trained checkpoint, useful for previewing in the 3DGRUT playground

For canonical training options, dataset preparation tips, and the full set of export flags, see the [3DGRUT README](https://github.com/nv-tlabs/3dgrut).

---

## Part 2 — Prepare the scene in Isaac Sim

This is the most hands-on part of the workflow. The 3DGUT splat is *visual only* — PhysX can't collide with Gaussian splats. You need to give the physics solver something to stand on (floor) and something to manipulate things on (table surface). The splat handles the look; the proxies handle the physics.

The mental model:

| Layer | Source | Role |
|---|---|---|
| Visual environment | NuRec USDZ | What cameras see |
| Collision proxies | Manual meshes you add | What PhysX sees |
| Manipulable object | Lightwheel SimReady asset | What the robot grabs |
| Robot | LW-BenchHub robot config | Embodiment + action space |
| Task logic + RL | `LiftObjFreeform` + LW-BenchHub | Success criteria, training |

### 2.1 Open the USDZ in Isaac Sim

Launch Isaac Sim 5.0+, new empty stage, **File → Import** the `convert.usdz` from your 3DGUT output. The splat appears as a NuRec volume prim. You can navigate with WASD or right-click drag.

The splat almost certainly won't be at the right orientation, height, or scale relative to world origin. That's the next two steps.

### 2.2 Re-anchor so the floor is at z=0

Apply a transform to the splat's root xform so:

- The visible floor of your scene sits at `z=0`
- The "front" of your manipulation area roughly faces `+x`
- The scene is at meter scale (a real-world 1m looks like 1m in the viewport)

**Quick scale check:** drop a default 1m cube next to a known-height object in your scene. If you captured a kitchen counter (real height ~0.9m) and your default cube is much shorter than the counter, your splat is over-scaled — apply a uniform scale to the splat's xform to compensate. If you captured a coffee table (real height ~0.5m) and the cube towers over it, you're under-scaled.

> **Practical note from my own captures:** 3DGUT-produced splats often come out at non-meter scale — mine was about 1.8x too big. Look at known-size objects (tables, chairs, a phone you placed in the scene) and scale until they look right. The scale factor will be uniform; just guess once, iterate, save when it looks right.

After scaling, you may need to translate the splat down (or up) to put the floor back at z=0, because USD scaling happens around the prim's origin which usually isn't at the floor.

### 2.3 Add a ground plane and table collider

**Ground plane:**

1. **Create → Physics → Ground Plane**
2. Position it at z=0 (default)
3. Make sure it covers the floor area of your splat (scale up if needed)

**Table collider:**

For a round table, use a thin cylinder:

1. **Create → Shapes → Cylinder**
2. Set scale to make it a thin disc matching your table — e.g., `(table_radius, table_radius, 0.01)`
3. Drag it up to sit on top of the visible table surface in the splat
4. **Physics → Collider** (not Rigid Body — static)

For a rectangular surface, use a thin box. For complex shapes, combine primitives or import a low-poly mesh with convex decomposition.

### 2.4 Make the proxies invisible-but-shadow-casting

For each proxy (ground plane and table collider):

1. **Property → Visibility → invisible**
2. **Property → Geometry → Matte Object → true**

Then register the proxies with the NuRec volume:

1. Select the NuRec volume prim (the splat root)
2. **Property → NuRec/Volume → Proxy** field
3. Click `+` and add the ground plane
4. Click `+` again and add the table collider

This is what makes shadows from objects above the table land correctly on the splat's rendered table surface. Without matte objects, objects look like they're floating.

### 2.5 Dress rehearsal

Drop a small test cube (default Create → Shapes → Cube, scaled to ~0.05) above your table proxy. Hit play. The cube should fall and rest on the proxy at the visible table surface level. If it falls through, the proxy isn't aligned. If it rests visibly above the splat's table, the proxy is too high.

If anything looks off, also:

- **Delete the DomeLight if Isaac Sim added one.** Multi-env training causes problems if every env has its own dome light. LW-BenchHub adds its own lighting at runtime via `LwScene.modify_env_cfg`, so the scene USD doesn't need its own.
- **Set the default prim.** Right-click the top-level xform in the Stage panel, "Set as Default Prim." LW-BenchHub looks for this when referencing your USD.

### 2.6 Save

`File → Save As → garden_scene_with_proxies.usd` (or whatever you want to call it). Use `.usd`, not `.usdz`. Place it somewhere stable — I keep mine at `~/lw_benchhub/assets/garden_scene_with_proxies.usd`.

Note that this `.usd` file *references* your `convert.usdz`. The reference is by absolute path. Either keep both files in known stable locations, or update the reference path in Isaac Sim if you move things later. If LW-BenchHub fails to load with a "Could not open asset" warning, that's the reference being broken.

---

## Part 3 — Wire into LW-BenchHub

This is the integration layer. Four small files to add (or copy from this repo's `code/` folder), plus one upstream decorator line.

### 3.1 The freeform task class

LW-BenchHub's `LiftObj` task anchors object placement to robocasa fixtures — there's a `self.counter` reference that determines where the cube spawns. Your NuRec scene has no counter fixture. The solution is a subclass that overrides fixture lookup and uses absolute world coordinates instead.

Copy [`code/lift_obj_freeform.py`](code/lift_obj_freeform.py) to:

```
~/lw_benchhub/lw_benchhub_tasks/lightwheel_robocasa_tasks/single_stage/lift_obj_freeform.py
```

The class is intentionally minimal:

```python
class LiftObjFreeform(LiftObj):
    task_name: str = "LiftObjFreeform"

    # ===== Edit these per scene =====
    CUBE_WORLD_POS = (-0.30, -0.06, 0.755)   # cube spawn (x, y, z)
    SUCCESS_HEIGHT_Z = 0.855                  # success threshold (z)
    ENV_SPACING = 50.0                        # multi-env grid spacing
    # ================================

    def __init__(self):
        super().__init__()
        self.fix_object_pose_cfg = {"object": {"pos": self.CUBE_WORLD_POS}}

    def _setup_kitchen_references(self, scene):
        # Skip robocasa fixture lookup
        self.fixture_refs = {}
        self.init_robot_base_ref = None

    def _get_obj_cfgs(self):
        return [dict(
            name="object",
            obj_groups="cube",
            graspable=True,
            placement=dict(
                size=(0.10, 0.10),
                pos=(self.CUBE_WORLD_POS[0], self.CUBE_WORLD_POS[1]),
                offset=(0, 0),
            ),
        )]

    def modify_env_cfg(self, env_cfg):
        env_cfg = super().modify_env_cfg(env_cfg)
        env_cfg.scene.env_spacing = self.ENV_SPACING
        return env_cfg

    def _check_success(self, env):
        if self.context.execute_mode == ExecuteMode.TRAIN:
            return torch.tensor([False], device=env.device).repeat(env.num_envs)
        object_height = env.scene["object"].data.root_pos_w[:, 2]
        return object_height >= self.SUCCESS_HEIGHT_Z
```

Three things to know:

- `CUBE_WORLD_POS` and `SUCCESS_HEIGHT_Z` are the values you'll edit per scene
- `_get_obj_cfgs` returns a placement that the LW-BenchHub sampler can validate against; `fix_object_pose_cfg` then overrides it with our exact world coords
- `ENV_SPACING = 50.0` is generous — splats often have stray Gaussians extending far beyond the visible scene. Adjust if you see neighbor scenes bleeding into your viewport during multi-env training

### 3.2 Register the new task as a Gym environment

Add a `gym.register` block to:

```
~/lw_benchhub/lw_benchhub_tasks/lightwheel_robocasa_tasks/single_stage/__init__.py
```

Right after the existing `Robocasa-Task-LiftObj` registration, add:

```python
gym.register(
    id="Robocasa-Task-LiftObjFreeform",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_obj_freeform:LiftObjFreeform",
    },
    disable_env_checker=True,
)
```

### 3.3 Enable the RL config on the new task

This is the one upstream edit. The RL configs are bound to specific task classes via the `@rl_on(task=...)` decorator using strict class equality — subclasses aren't picked up automatically.

Open:

```
~/lw_benchhub/lw_benchhub_rl/lift_obj/lift_obj.py
```

Add an import near the top, alongside the existing `LiftObj` import:

```python
from lw_benchhub_tasks.lightwheel_robocasa_tasks.single_stage.lift_obj_freeform import LiftObjFreeform
```

Then find each `@rl_on(task=LiftObj)` decorator in the file (there are usually two) and stack a freeform decorator below it:

```python
@rl_on(task=LiftObj)
@rl_on(task=LiftObjFreeform)  # ← add this line
class LeRobotLiftObjStateRL(...):
```

Stacked decorators register the RL config against both task types. The same trained PPO checkpoint can now be evaluated in both kitchen and freeform tasks.

### 3.4 The CSV pose row

Copy [`code/nurec_scenes.csv`](code/nurec_scenes.csv) to a new file under LW-BenchHub's configs:

```
~/lw_benchhub/configs/layout_task_mapping/nurec_scenes.csv
```

LW-BenchHub's `CSVLoader` globs all `*.csv` files under `configs/`, so a new file is auto-discovered. No edits to the existing `layout_task_mapping.csv` required.

The contents:

```csv
robot,layout,task,init_robot_base_pos,init_robot_base_ori,object_init_offset
LeRobot-RL,/home/jonathan/lw_benchhub/assets/garden_scene_with_proxies.usd,LiftObjFreeform,"[-0.18, -0.25, 0.73]","[0.0, 0.0, 3.70]","[0.0, 0.0]"
```

Edit per scene:

- `layout`: the absolute filesystem path to your scene USD (must match exactly what you'll put in the YAML below)
- `init_robot_base_pos`: where the LeRobot mounts on your table — `[x, y, z]` in world coords
- `init_robot_base_ori`: Euler angles in radians — `[roll, pitch, yaw]`. Only yaw usually matters; rotate to face the cube
- `object_init_offset`: unused for our freeform task (the cube pose is set via `CUBE_WORLD_POS` in the Python class), but the CSV schema requires it

The `robot` column matches the embodiment's `self.name` string. For the default RL LeRobot, that's `LeRobot-RL`. Other variants in the LW-BenchHub codebase: `LeRobot`, `LeRobot100-RL`, `LeRobot-AbsJointGripper-RL`. The string match is exact.

### 3.5 The training and eval YAMLs

Copy [`code/lerobot_liftobj_freeform_state.yaml`](code/lerobot_liftobj_freeform_state.yaml) to:

```
~/lw_benchhub/configs/rl/skrl/lerobot_liftobj_freeform_state.yaml
```

Contents:

```yaml
_base_:
  - rl_base
task: LiftObjFreeform
robot: LeRobot-RL
layout: /home/jonathan/lw_benchhub/assets/garden_scene_with_proxies.usd
rl: LeRobotLiftObjStateRL
num_envs: 10
usd_simplify: false
enable_cameras: false
```

Notes:

- `task` matches the `task_name` class variable on `LiftObjFreeform`
- `layout` is your USD path — must byte-match the CSV row's `layout` column
- `rl: LeRobotLiftObjStateRL` reuses the existing PPO config (no changes needed because we stacked the `@rl_on` decorator)
- `usd_simplify: false` is important — the simplify pass is tuned for robocasa geometry and can corrupt a NuRec volume prim

For evaluation, copy [`code/lerobot_liftobj_freeform_state_play.yaml`](code/lerobot_liftobj_freeform_state_play.yaml):

```yaml
_base_:
  - lerobot_liftobj_freeform_state
num_envs: 1
checkpoint: /path/to/your/best_agent.pt
```

The `_base_` line inherits everything from the training config. Override `num_envs` for the eval run, and point `checkpoint` at any compatible PPO checkpoint — including one trained on a robocasa kitchen.

### 3.6 Quick sanity check

Confirm the registration worked before launching the full pipeline:

```bash
cd ~/lw_benchhub
conda activate lw_benchhub
python -c "
import lw_benchhub_tasks.lightwheel_robocasa_tasks.single_stage
import gymnasium as gym
print('Registered:', 'Robocasa-Task-LiftObjFreeform' in gym.registry)
"
```

You want to see `Registered: True`. If you see an error, paste it in an issue and I'll help debug.

---

## Part 4 — Run training

```bash
cd ~/lw_benchhub
conda activate lw_benchhub
python ./lw_benchhub/scripts/rl/train.py --task_config lerobot_liftobj_freeform_state
```

First boot is slow — Isaac Sim takes a couple of minutes to come up.

You should see, in order:

1. Isaac Sim startup messages (driver detection, plugin loads)
2. `[INFO]: Parsing configuration from: ... lift_obj_freeform:LiftObjFreeform`
3. `load floorplan usd...[backend->robocasa] | [scene->usd]` — confirms your USD is being loaded directly
4. `[CSV Match] LeRobot-RL | /path/to/your.usd | LiftObjFreeform | Init Pos:[...]` — confirms CSV lookup worked
5. `Sampled object: BuildingBlock003 from lightwheel` — Lightwheel SDK fetched a cube
6. `Placed object 'object' successfully` — cube placement passed validation
7. Viewport opens with your scene and a `num_envs` grid of robots

After that, PPO starts iterating. With `num_envs: 10`, expect a couple of thousand steps per second on a modern RTX card.

---

## Part 5 — Evaluate with an existing checkpoint

This is what I think is the most interesting part of the integration. If you have a PPO checkpoint trained on the original `LiftObj` task in robocasa kitchens, you can drop it directly into your NuRec scene and see what it does:

```bash
python ./lw_benchhub/scripts/rl/play.py --task_config lerobot_liftobj_freeform_state_play
```

The policy was trained on a perfectly authored synthetic kitchen counter. It's now seeing a captured real-world patio table at different coordinates, different scale relative to the cube, different (state-based) observation distribution. Whether it succeeds, fails, or fails *interestingly* tells you something about what the policy actually learned.

This is the actual value proposition of the integration: **measuring sim-to-real transfer gaps using captured environments, without rebuilding scenes by hand**.

---

## Gotchas

A list of things that bit me during development, ordered roughly by how much time they cost.

**Splat scale is arbitrary.** 3DGUT's `apply_normalizing_transform` centers and roughly scales the scene, but the result isn't metric. Drop a default 1m cube next to known-size objects in your splat to check. If everything looks ~1.8x bigger than reality, apply a 0.55 scale to the splat's root xform. Iterate by eye until proportions look right.

**Scaling shifts the floor.** USD prims scale around their origin, which usually isn't the floor. After scaling the splat, you'll need to translate it back down so the visible floor lines up with your z=0 ground plane.

**The `_enabled.usd` cache.** LW-BenchHub processes your scene USD and exports `<name>_enabled.usd` as a cache. If you re-edit the source USD, the cache is stale. Delete `~/lw_benchhub/assets/<name>_enabled.usd` after any USD edit — it'll be regenerated.

**Broken USD references.** Your scene USD references `convert.usdz` by absolute path. If you move the splat file, the reference breaks silently and the splat stops appearing. Check `~/lw_benchhub/assets/` for both files, or update the reference path in Isaac Sim.

**Multi-env DomeLights overlap.** If your scene USD has a DomeLight baked in, every parallel env will instantiate one and you'll get nuked-out lighting. Delete the DomeLight from the scene USD; LW-BenchHub adds appropriate lighting at runtime.

**Multi-env splat bleed-through.** Default `env_spacing` of 30m is enough for kitchens but not for splats with far-flung stray Gaussians. The freeform task class sets `ENV_SPACING = 50.0` to compensate. Crank higher if you still see overlap.

**`fixture_refs` not initialized.** If you write your own task override that skips `_setup_kitchen_references`, remember to initialize `self.fixture_refs = {}` — the base class's `_init_ref_fixtures` iterates over it and crashes if missing.

**The placement sampler needs *something*.** Returning an empty placement from `_get_obj_cfgs` causes the sampler to fail validation in an infinite retry loop. Return a small but non-degenerate placement (e.g. a 10cm × 10cm area at your target XY) so the sampler completes; `fix_object_pose_cfg` discards its output afterward.

**`@rl_on` uses strict class equality.** The RL config decorator does `type(task) in [...]`, not `isinstance(task, ...)`. Subclasses don't match unless you stack a second decorator with the subclass explicitly. This is the one upstream edit the integration requires.

**State-based vs visual observations.** The freeform task config uses `enable_cameras: false`, which means observations are state vectors (joint positions, cube pose). The splat doesn't affect those observations. Switching to a vision variant — where observations include rendered camera images — is where the photoreal scene actually shows up in training. That's a separate experiment.

---

## What's next

Things this integration enables that I haven't built out yet:

- **Visual policies on photoreal scenes.** Switch to `enable_cameras: true` and a camera-based observation config. Now the splat is the input the policy sees.
- **Multi-scene domain randomization.** Train on N captured scenes, eval on a held-out one. Tests generalization in a way synthetic-only training can't.
- **Other LW-BenchHub tasks beyond LiftObj.** The same `LiftObjFreeform` pattern (skip fixture lookup + override placement) extends to any single-stage manipulation task. PickPlace and stacking should be straightforward.
- **Stereo NuRec.** Stereo captures give metric-scale splats out of the box, removing the rescaling step. See the [NuRec stereo workflow](https://docs.nvidia.com/nurec/robotics/neural_reconstruction_stereo.html).

---

## Resources

- [LW-BenchHub](https://github.com/LightwheelAI/LW-BenchHub) · [LW-BenchHub docs](https://docs.lightwheel.net/lw_benchhub/)
- [3DGRUT](https://github.com/nv-tlabs/3dgrut)
- [NuRec mono-camera workflow](https://docs.nvidia.com/nurec/robotics/neural_reconstruction_mono.html)
- [Isaac Sim NuRec asset docs](https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_nurec.html)
- [Lightwheel SimReady assets](https://github.com/LightwheelAI/Lightwheel-simready-asset)

---

## A video walkthrough is coming

I'll be recording a video that follows the same structure as this guide — phone capture through training rollout — and embedding it here when it's done. In the meantime, if you hit a snag, open an issue and I'll work through it with you.

## License

Apache 2.0, matching LW-BenchHub.
