"""GIFs, posters and the 'how a frame is built' figure from rendered frames.

    python scripts/make_media.py gifs
    python scripts/make_media.py method
    python scripts/make_media.py carry
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
MEDIA = ROOT / "media"

# excerpt of each walk used for the GIFs and loops: (first frame, seconds)
EXCERPTS = {"leake": (300, 6.0), "linden": (330, 6.0)}
LOOP_SECONDS = 10.0
SRC_FPS = 25


def ffmpeg(*args) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, args)], check=True)


def gif(src_dir: Path, out: Path, start: int, seconds: float, width: int, fps: int = 8,
        colors: int = 96, crop: str | None = None) -> None:
    """Photoreal footage with a moving camera compresses badly as GIF: no dither and
    a small palette keep a 6 s three-view clip near 7 MB (10 s as MP4 is <1 MB, which
    is why the web page uses the -loop.mp4 files)."""
    pre = f"crop={crop}," if crop else ""
    vf = (f"{pre}fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
          f"[a]palettegen=max_colors={colors}:stats_mode=full[p];[b][p]paletteuse=dither=none")
    ffmpeg("-start_number", start, "-framerate", SRC_FPS, "-i", src_dir / "%05d.png",
           "-frames:v", int(seconds * SRC_FPS), "-vf", vf, out)
    print(f"wrote {out.name} ({out.stat().st_size / 1e6:.2f} MB)")


def gifs():
    for name, (start, secs) in EXCERPTS.items():
        frames = ROOT / "data" / name / "frames"
        gif(frames, MEDIA / f"{name}.gif", start, secs, width=720)
        loop = MEDIA / f"{name}-loop.mp4"
        ffmpeg("-start_number", start, "-framerate", SRC_FPS, "-i", frames / "%05d.png",
               "-frames:v", int(LOOP_SECONDS * SRC_FPS), "-c:v", "libx264", "-crf", 26,
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", loop)
        print(f"wrote {loop.name} ({loop.stat().st_size / 1e6:.2f} MB)")
        iio.imwrite(MEDIA / f"{name}-poster.jpg", iio.imread(frames / f"{start:05d}.png"), quality=88)


def carry():
    """Pick and place excerpts of the warehouse carry demo."""
    import numpy as np
    z = np.load(ROOT / "data" / "linden" / "carry_demo.npz", allow_pickle=True)
    labels = [str(x) for x in z["labels"]]
    frames = ROOT / "data" / "linden" / "carry_frames"
    for name, phase, lead in [("pick", "reach", 15), ("place", "lower", 60)]:
        start = labels.index(phase) - lead
        gif(frames, MEDIA / f"linden_carry_{name}.gif", start, 6.0, width=720)
        loop = MEDIA / f"linden_carry_{name}-loop.mp4"
        ffmpeg("-start_number", start, "-framerate", SRC_FPS, "-i", frames / "%05d.png",
               "-frames:v", int(LOOP_SECONDS * SRC_FPS), "-c:v", "libx264", "-crf", 26,
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", loop)
        print(f"wrote {loop.name} ({loop.stat().st_size / 1e6:.2f} MB)")
        iio.imwrite(MEDIA / f"linden_carry_{name}-poster.jpg", iio.imread(frames / f"{start + 40:05d}.png"), quality=88)


def method(name: str = "linden", frame: int = 470):
    """Three stages of one chase-camera frame: splat alone, MuJoCo passes, composite."""
    from g1places.composite import Compositor
    from g1places.scenes import SCENES
    from g1places.sim import build_model
    from g1places.splat import SplatScene, intrinsics
    from render import chase_poses, label, set_chase

    sc = SCENES[name]
    qpos = np.load(sc.dir / "walk.npz")["qpos"]
    model = build_model([])
    model.cam_fovy[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "chase")] = 60.0
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[frame]
    set_chase(model, data, chase_poses(qpos)[frame])
    mujoco.mj_forward(model, data)
    W, H = 640, 480
    comp = Compositor(model, sc, SplatScene(sc.dir / "splats.npz"), W, H, robot_gain=sc.robot_gain)
    c2w = sc.world_to_file_pose(comp.camera_pose_cv(data, "chase"))
    splat_rgb, _ = comp.splats.render(c2w, intrinsics(W, H, 60.0), W, H)
    mj = comp._mj(data, "chase", comp.opt_robot_floor)
    small = mj.reshape(H, 2, W, 2, 3).mean(axis=(1, 3)).astype(np.uint8)
    final = comp.render(data, "chase")["rgb"]
    panels = [label((splat_rgb * 255).astype(np.uint8), "1  Gaussian splat (metal-gauss)"),
              label(small, "2  MuJoCo: robot + floor shadow"),
              label((final * 255).astype(np.uint8), "3  composited by depth")]
    iio.imwrite(MEDIA / "how_a_frame_is_built.jpg", np.concatenate(panels, 1), quality=90)
    print("wrote", MEDIA / "how_a_frame_is_built.jpg")


if __name__ == "__main__":
    {"gifs": gifs, "method": method, "carry": carry}[sys.argv[1]]()
