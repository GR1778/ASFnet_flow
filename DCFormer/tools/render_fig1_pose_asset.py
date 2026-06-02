import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BONES_I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
BONES_J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
LEFT_BONE = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)

H36M_CAMERA_ORIENTATION = {
    ("S11", 2): np.array([0.6189449429512024, -0.7600917220115662, -0.15300633013248444, 0.1255258321762085], dtype=np.float32),
}


def qrot(q, v):
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    qvec = q[1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (q[0] * uv + uuv)


def camera_to_world(points, subject, camera):
    q = H36M_CAMERA_ORIENTATION[(subject, camera)]
    return qrot(q, np.asarray(points, dtype=np.float32).reshape(17, 3))


def camera_to_paper_view(points, mirror_x=False, mirror_depth=False):
    """Map H36M camera coordinates to a paper-friendly upright display."""
    points = np.asarray(points, dtype=np.float32).reshape(17, 3)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x = -x if mirror_x else x
    z = -z if mirror_depth else z
    h = -y
    h = h - h.min()
    return np.stack([x, z, h], axis=1)


def set_equal_axes(ax, points, xy_pad=0.14, z_pad=0.12):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    xy_span = max(maxs[0] - mins[0], maxs[1] - mins[1], 0.85)
    ax.set_xlim3d(center[0] - xy_span / 2 - xy_pad, center[0] + xy_span / 2 + xy_pad)
    ax.set_ylim3d(center[1] - xy_span / 2 - xy_pad, center[1] + xy_span / 2 + xy_pad)
    ax.set_zlim3d(max(0.0, mins[2] - 0.03), maxs[2] + z_pad)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.04))
    except Exception:
        ax.set_aspect("auto")


def render_pose(
    points,
    output,
    elev=15.0,
    azim=110.0,
    line_width=1.75,
    mirror_x=False,
    mirror_depth=False,
    official_world=False,
    subject="S11",
    camera=2,
):
    if official_world:
        points = camera_to_world(points, subject, camera)
        points[:, 2] -= points[:, 2].min()
    else:
        points = camera_to_paper_view(points, mirror_x=mirror_x, mirror_depth=mirror_depth)

    fig = plt.figure(figsize=(2.55, 1.95), dpi=360)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    left_color = "#1c55ff"
    right_color = "#ff2525"
    for idx, (start, end) in enumerate(zip(BONES_I, BONES_J)):
        color = left_color if LEFT_BONE[idx] else right_color
        ax.plot(
            [points[start, 0], points[end, 0]],
            [points[start, 1], points[end, 1]],
            [points[start, 2], points[end, 2]],
            color=color,
            lw=line_width,
            solid_capstyle="round",
        )

    set_equal_axes(ax, points)

    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis._axinfo["grid"]["color"] = (0.76, 0.76, 0.76, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.45
        axis._axinfo["axisline"]["color"] = (0.66, 0.66, 0.66, 1.0)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(axis="both", which="both", length=0, pad=-4)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render a CAPF/ASFNet-style 3D pose asset for Fig. 1.")
    parser.add_argument("--pose", required=True, help="Path to a 17x3 .npy pose in H36M camera coordinates.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--elev", type=float, default=15.0)
    parser.add_argument("--azim", type=float, default=110.0)
    parser.add_argument("--mirror-x", action="store_true")
    parser.add_argument("--mirror-depth", action="store_true")
    parser.add_argument("--official-world", action="store_true")
    parser.add_argument("--subject", default="S11")
    parser.add_argument("--camera", type=int, default=2)
    args = parser.parse_args()

    pose = np.load(args.pose)
    render_pose(
        pose,
        args.output,
        elev=args.elev,
        azim=args.azim,
        mirror_x=args.mirror_x,
        mirror_depth=args.mirror_depth,
        official_world=args.official_world,
        subject=args.subject,
        camera=args.camera,
    )


if __name__ == "__main__":
    main()
