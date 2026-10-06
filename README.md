# A Unitree G1 walking through real places, on a laptop

A Unitree G1 humanoid walks through two real-world 3D Gaussian-splat captures: a graffiti tunnel in London and a pallet-rack warehouse in New Jersey. It is recorded from three synchronised cameras: third person, head-camera RGB and head-camera depth.

The setup this copies is NVIDIA Isaac Sim, Isaac Lab, an RTX GPU and Niantic Spatial's Places Library ([Tim Martin's post](https://www.linkedin.com/posts/tim-martin-176471250_nvidiaomniverse-isaacsim-isaaclab-ugcPost-7512778691396018176-J65_/)). This version runs on an Apple-silicon MacBook with no NVIDIA GPU and no cloud. It costs $0:

- **MuJoCo** runs the physics and draws the robot.
- **[metal-gauss](https://github.com/nandometzger/metal-gauss)** draws the scene on the Mac's GPU.
- **Unitree's own pretrained walking policy** moves the legs.

![Leake Street Arches: third person, head RGB, head depth](media/leake.gif)

![Linden warehouse: third person, head RGB, head depth](media/linden.gif)

Full walks: [`media/leake.mp4`](media/leake.mp4), [`media/linden.mp4`](media/linden.mp4) (38 s each).

## Results

| | Leake Street Arches, London | Warehouse, Linden NJ |
|---|---|---|
| Gaussians in the capture | 10,581,232 | 12,774,981 |
| Collision mesh | 0.5 M triangles | 26.7 M triangles |
| Walked | 21.8 m in 38.4 s | 21.4 m in 37.7 s |
| Falls / obstacle contacts | 0 / 0 | 0 / 0 |
| Distance from the planned path, mean / max | 0.06 / 0.12 m | 0.07 / 0.13 m |
| Scale correction needed | **×5.4** (estimated, see below) | none |
| Up axis correction needed | none | **Y-up → Z-up** |
| Floor tilt removed | 0.90° | 0.21° |

Rendering three 480×360 views takes **0.85 s per frame** on an M5 MacBook Pro (32 GB). Each frame needs two splat colour renders, two splat depth renders and ten MuJoCo passes. That works out to about 14 minutes per 38-second walk.

![Top-down maps of both walks](media/walk_maps.png)

## How a frame is built

![Splat render, MuJoCo robot pass, composite](media/how_a_frame_is_built.jpg)

1. **Scene.** The USDZ's `.nurec` file is a gzip-compressed msgpack 3DGUT checkpoint. It holds fp16 positions, rotations, log-scales, density logits and degree-3 spherical harmonics. [`g1places/scene.py`](g1places/scene.py) decodes it, and `metal-gauss` rasterises it with its fused Metal kernels.
2. **Robot.** MuJoCo renders the G1 with the scene hidden: colour, depth and a segmentation mask, at 2× resolution then box-filtered for anti-aliased edges.
3. **Shadow.** MuJoCo renders a grey floor plane under a shadow-casting light, once with the robot and once without. The ratio between the two darkens the splat floor where the feet are.
4. **Composite.** A robot pixel wins wherever the robot is closer to the camera than the splat surface.
5. **Depth camera.** Each Gaussian's camera-space z is written into its SH DC coefficient, so the same fused kernel rasterises depth. That's 4× faster than metal-gauss's explicit-colour path (0.27 vs 1.11 s) and agrees with it to the millimetre.

The legs are driven by `deploy/pre_train/g1/motion.pt` from [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym), with the observation layout, gains and 50 Hz control unchanged. On top of it sits a path follower: it commands forward speed (0.6 m/s) and steers toward a point 1.2 m ahead on an A* path. The path is planned on an occupancy grid built from the collision mesh. The cameras are recorded but not used for control: this is a rendering demo, not a perception policy.

## What the sample files needed

The two free samples don't agree with each other or with their own metadata. Each issue below would have broken a naive import.

**Leake Street is about 5× too small.** In the file, the tunnel is 1.48 units wall to wall and about 1.3 units floor to ceiling, so the 1.3 m robot would barely fit. The tunnel is published as 8 m wide, which gives a scale of 5.4. A second check agrees: the reconstruction origin (most likely the first camera pose) sits 0.33 units above the floor, which is 1.8 m at that scale, a handheld 360-camera height. Linden's origin sits 1.79 m above its floor. Treat the ×5.4 as an estimate, good to perhaps ±20%. The mesh layer also declares centimetres (`metersPerUnit = 0.01`), but its numbers are in the same units as the splat.

**Linden is Y-up inside a Z-up stage.** Both samples author an identity transform on the splat. Leake's splat is Z-up and Linden's is Y-up, so no single convention reads both correctly. Leake's mesh carries a Y→Z rotation; Linden's carries none.

**Neither floor is level.** A robust plane fit to the floor gives a 0.90° tilt for Leake (4.4 cm residual) and 0.21° for Linden (0.5 cm). My first obstacle map for Leake measured height above z=0. The floor rose 14 cm along the walk, so the map turned half the tunnel floor into "walls". Checking those walls against the splat, which showed an empty tunnel there, is what caught it. Each world frame is now levelled from the fit ([`fit_floor`](g1places/collision.py)).

**MuJoCo collides with the convex hull of a mesh.** Loaded as one mesh, the scan becomes a solid lump with the robot inside it. Instead, only geometry 0.15–1.6 m above the levelled floor is kept. It's rasterised to a 10 cm grid and merged into boxes (39 near the Leake path, 419 near the Linden one). The floor is a plane, because the policy was trained on flat ground.

**The splats are never transformed.** Rotating 10 M Gaussians would mean rotating their spherical harmonics too. Instead, each world camera pose is mapped into the file frame (`Scene.world_to_file_pose`), and splat depth is scaled back to metres.

## Limitations

- The walking policy is blind and was trained on flat ground. Steps, ramps and anything under 15 cm aren't modelled.
- Collision is 2.5D: no overhangs, no tables to walk under.
- The robot's brightness is matched to each capture by hand (`robot_gain`), not relit from it.
- Splat depth is alpha-weighted expected depth. It's smooth at edges and floaters, unlike a real stereo depth camera.
- Leake's scale is inferred from one published width plus a camera-height check.
- One walk per scene. No randomisation, no repeated seeds.

## Run it

```bash
scripts/setup.sh        # venv + unitree_rl_gym at a pinned commit (G1 model + policy)

# the two free samples: https://www.nianticspatial.com/en/embodied-ai (download form)
.venv/bin/python scripts/prepare.py leake  ~/Downloads/LeakeStreetArches.usdz
.venv/bin/python scripts/prepare.py linden ~/Downloads/Linden_Warehouse_Pallet_Racks.usdz

.venv/bin/python scripts/simulate.py leake          # plan + walk -> data/leake/walk.npz
.venv/bin/python scripts/render.py leake --range 0 480
.venv/bin/python scripts/render.py leake --range 480 960
.venv/bin/python scripts/render.py leake --encode   # -> media/leake.mp4
.venv/bin/python scripts/figures.py                 # -> media/walk_maps.png
.venv/bin/python scripts/make_media.py gifs         # -> GIFs, loops, posters
.venv/bin/python scripts/make_media.py method       # -> media/how_a_frame_is_built.jpg
.venv/bin/python scripts/check_sh_layout.py linden  # SH coefficient order check
.venv/bin/python -m pytest tests                    # geometry tests, no GPU or data needed
```

Render in foreground chunks. macOS throttled the same GPU job about 7× when it ran as a background process (6 s/frame vs 0.85 s/frame).

## Credits and licences

- **Scenes:** Niantic Spatial, Places Library free samples (Leake Street Arches; Linden Warehouse Pallet Racks). The data isn't redistributed here; download it from Niantic. Renders of the scenes in `media/` are shown for demonstration.
- **G1 model and walking policy:** [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym), BSD-3-Clause, fetched by `scripts/setup.sh`.
- **Splat rasteriser:** [metal-gauss](https://github.com/nandometzger/metal-gauss), MIT.
- **Physics:** [MuJoCo](https://github.com/google-deepmind/mujoco), Apache-2.0.
- **Idea:** Tim Martin's Isaac Sim demo of the same pipeline.

Code in this repository: MIT.
