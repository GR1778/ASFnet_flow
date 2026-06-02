import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PARENTS_17 = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
RIGHT_JOINTS_17 = {1, 2, 3, 14, 15, 16}
LEFT_JOINTS_17 = {4, 5, 6, 11, 12, 13}

H36M_CAMERAS = {
    1: {
        "orientation": [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
        "azimuth": 70,
    },
    2: {
        "orientation": [0.6189449429512024, -0.7600917220115662, -0.15300633013248444, 0.1255258321762085],
        "azimuth": -70,
    },
    3: {
        "orientation": [0.14651472866535187, -0.14647851884365082, 0.7653023600578308, -0.6094175577163696],
        "azimuth": 110,
    },
    4: {
        "orientation": [0.5834008455276489, -0.7853162288665771, 0.14548823237419128, -0.14749594032764435],
        "azimuth": -110,
    },
}


def qrot(q, v):
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    qvec = q[1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (q[0] * uv + uuv)


def camera_to_world(points, camera_id):
    camera = H36M_CAMERAS[camera_id]
    q = np.asarray(camera["orientation"], dtype=np.float32)
    points = np.asarray(points, dtype=np.float32).reshape(17, 3)
    return qrot(q, points)


def prepare_pose(points, camera_id):
    points = camera_to_world(points, camera_id)
    points[:, 2] -= np.min(points[:, 2])
    return points


def draw_2d(ax, image_path, joints2d_path=None, title=None):
    image = Image.open(image_path).convert("RGB")
    ax.imshow(image)
    if title:
        ax.set_title(title)
    ax.set_axis_off()

    if joints2d_path is None:
        return

    joints = np.load(joints2d_path).reshape(17, 2)
    for joint, parent in enumerate(PARENTS_17):
        if parent == -1:
            continue
        ax.plot(
            [joints[joint, 0], joints[parent, 0]],
            [joints[joint, 1], joints[parent, 1]],
            color="pink",
            linewidth=1.4,
        )
    colors = ["red" if idx in RIGHT_JOINTS_17 else "black" for idx in range(17)]
    ax.scatter(joints[:, 0], joints[:, 1], s=10, c=colors, edgecolors="white", linewidths=0.35, zorder=3)


def draw_3d(ax, pose, title, azimuth):
    radius = 1.7
    root = pose[0]
    ax.view_init(elev=15.0, azim=azimuth)
    ax.set_xlim3d([-radius / 2 + root[0], radius / 2 + root[0]])
    ax.set_ylim3d([-radius / 2 + root[1], radius / 2 + root[1]])
    ax.set_zlim3d([0, radius])
    try:
        ax.set_aspect("equal")
    except NotImplementedError:
        ax.set_aspect("auto")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(axis="both", which="both", length=0, pad=-4)
    if title:
        ax.set_title(title)

    for joint, parent in enumerate(PARENTS_17):
        if parent == -1:
            continue
        color = "red" if joint in RIGHT_JOINTS_17 else "black"
        ax.plot(
            [pose[joint, 0], pose[parent, 0]],
            [pose[joint, 1], pose[parent, 1]],
            [pose[joint, 2], pose[parent, 2]],
            zdir="z",
            c=color,
            linewidth=2.0,
            solid_capstyle="round",
        )


def save_single_panel(fig, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_split(args, pred, gt, azimuth):
    split_dir = Path(args.split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(args.size, args.size), dpi=args.dpi)
    ax = fig.add_subplot(111)
    draw_2d(ax, args.image, args.joints2d, None if args.no_titles else "Input")
    save_single_panel(fig, split_dir / f"{args.basename}_input2d.png")

    fig = plt.figure(figsize=(args.size, args.size), dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")
    draw_3d(ax, pred, None if args.no_titles else args.pred_title, azimuth)
    save_single_panel(fig, split_dir / f"{args.basename}_pred3d.png")

    if gt is not None:
        fig = plt.figure(figsize=(args.size, args.size), dpi=args.dpi)
        ax = fig.add_subplot(111, projection="3d")
        draw_3d(ax, gt, None if args.no_titles else args.gt_title, azimuth)
        save_single_panel(fig, split_dir / f"{args.basename}_gt3d.png")


def render(args):
    pred = prepare_pose(np.load(args.pred3d), args.camera)
    gt = prepare_pose(np.load(args.gt3d), args.camera) if args.gt3d else None
    azimuth = H36M_CAMERAS[args.camera]["azimuth"]

    if args.split_dir:
        render_split(args, pred, gt, azimuth)

    ncols = 3 if gt is not None else 2
    fig = plt.figure(figsize=(args.size * ncols, args.size), dpi=args.dpi)
    ax_in = fig.add_subplot(1, ncols, 1)
    draw_2d(ax_in, args.image, args.joints2d, None if args.no_titles else "Input")

    ax_pred = fig.add_subplot(1, ncols, 2, projection="3d")
    draw_3d(ax_pred, pred, None if args.no_titles else args.pred_title, azimuth)

    if gt is not None:
        ax_gt = fig.add_subplot(1, ncols, 3, projection="3d")
        draw_3d(ax_gt, gt, None if args.no_titles else args.gt_title, azimuth)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render a VideoPose3D/CAPF-style H36M still from exported arrays.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--joints2d")
    parser.add_argument("--pred3d", required=True)
    parser.add_argument("--gt3d")
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera", type=int, default=2, choices=sorted(H36M_CAMERAS))
    parser.add_argument("--pred-title", default="Reconstruction")
    parser.add_argument("--gt-title", default="Ground Truth")
    parser.add_argument("--no-titles", action="store_true")
    parser.add_argument("--split-dir")
    parser.add_argument("--basename", default="h36m_official")
    parser.add_argument("--size", type=float, default=2.6)
    parser.add_argument("--dpi", type=int, default=260)
    args = parser.parse_args()
    render(args)


if __name__ == "__main__":
    main()
