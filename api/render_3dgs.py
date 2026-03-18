"""
3D Gaussian Splatting 渲染完整实现
支持三种相机输入方式：look-at / c2w矩阵 / 批量JSON

依赖: torch, gsplat, numpy, plyfile, Pillow
安装: pip install gsplat plyfile Pillow numpy torch

python api/render_3dgs.py --ply dataset/scene/point_cloud.ply --eye 0.5 0 0.3 --target 0 0 0 --up 0 1.0 0 --out test.png

用法示例：
  # look-at 视角
python api/render_3dgs.py --ply dataset/scene/point_cloud.ply   --eye 0.31 0.5 -0.2 --target  0.3 0.35 -0.2 --up 0 1.0 0     --output frame.png

  # c2w 矩阵（行优先16个值）
python api/render_3dgs.py --ply dataset/scene/point_cloud.ply  --c2w 1 0 0 0  0 1 0 1.5  0 0 1 3.0  0 0 0 1  --output frame.png

  # 批量 JSON
python api/render_3dgs.py --ply dataset/scene/point_cloud.ply    --cameras_json cameras.json --output output/
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData

from gsplat.rendering import rasterization


# ══════════════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Gaussians:
    """存储 3DGS 模型的全部属性（均在 GPU 上）"""
    means:     torch.Tensor   # (N, 3)
    quats:     torch.Tensor   # (N, 4)  已归一化
    scales:    torch.Tensor   # (N, 3)  对数尺度
    opacities: torch.Tensor   # (N,)    logit
    colors:    torch.Tensor   # (N, 3)  RGB [0,1]


@dataclass
class CameraPose:
    """相机位姿"""
    view_matrix: torch.Tensor   # (1, 4, 4) 世界→相机
    K:           torch.Tensor   # (1, 3, 3) 内参矩阵
    position:    np.ndarray     # (3,)  调试用
    yaw:         float          # 偏航角（弧度），look-at/yaw 模式下有效


# ══════════════════════════════════════════════════════════════════════════════
# 1. 加载 PLY
# ══════════════════════════════════════════════════════════════════════════════

def load_ply(ply_path: str, device: str = "cuda") -> Gaussians:
    """
    从标准 3DGS 训练输出的 PLY 文件加载高斯属性。
    属性列：x y z / rot_0..3 / scale_0..2 / opacity / f_dc_0..2
    """
    path = Path(ply_path)
    if not path.exists():
        raise FileNotFoundError(f"PLY 文件不存在: {ply_path}")

    ply = PlyData.read(str(path))
    vtx = ply["vertex"]

    def col(name):
        return torch.tensor(np.array(vtx[name], dtype=np.float32), device=device)

    means = torch.stack([col("x"), col("y"), col("z")], dim=-1)

    quats = F.normalize(
        torch.stack([col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")], dim=-1),
        dim=-1
    )

    # 必须 exp
    scales = torch.exp(
        torch.stack([col("scale_0"), col("scale_1"), col("scale_2")], dim=-1)
    )

    # 必须 sigmoid
    opacities = torch.sigmoid(col("opacity"))

    SH_C0 = 0.28209479177387814
    f_dc = torch.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], dim=-1)

    colors = (0.5 + SH_C0 * f_dc).clamp(0.0, 1.0)

    print(f"[load_ply] 加载了 {means.shape[0]:,} 个高斯，来自 {ply_path}")
    return Gaussians(means=means, quats=quats, scales=scales,
                     opacities=opacities, colors=colors)


def scene_bounds(gaussians: Gaussians):
    """返回场景的 (min_xyz, max_xyz, center)，numpy float64。"""
    means_np = gaussians.means.cpu().numpy()
    return means_np.min(axis=0), means_np.max(axis=0), means_np.mean(axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 内参矩阵工具
# ══════════════════════════════════════════════════════════════════════════════

def make_K(fov_deg: float, W: int, H: int, device: str) -> torch.Tensor:
    """由水平 FOV 构造 3×3 内参矩阵，返回 (1, 3, 3)。"""
    fov   = math.radians(fov_deg)
    fx    = 0.5 * W / math.tan(fov / 2.0)
    fy    = fx
    K = torch.tensor([
        [fx,  0,  W / 2.0],
        [ 0, fy,  H / 2.0],
        [ 0,  0,  1.0    ],
    ], dtype=torch.float32, device=device).unsqueeze(0)
    return K


# ══════════════════════════════════════════════════════════════════════════════
# 3. 三种相机构造方式
# ══════════════════════════════════════════════════════════════════════════════

def pose_from_lookat(
    eye:    np.ndarray,
    target: np.ndarray,
    up:     np.ndarray,
    K:      torch.Tensor,
    device: str,
) -> CameraPose:
    """
    look-at → CameraPose
    --eye / --target / --up 参数对应此函数。
    """
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        raise ValueError("eye 和 target 不能是同一个点")
    forward /= norm

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        raise ValueError("forward 与 up 平行，请调整 --up 方向")
    right /= right_norm

    up_real = np.cross(right, forward)

    # c2w：列 = right, up_real, -forward, eye
    c2w      = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = up_real
    c2w[:3, 2] = -forward
    c2w[:3, 3] = eye

    view_mat = np.linalg.inv(c2w)

    # 从 forward 和 right 估算 yaw（绕 Y 轴）
    yaw = math.atan2(forward[0], forward[2])

    return CameraPose(
        view_matrix=torch.tensor(view_mat, dtype=torch.float32).unsqueeze(0).to(device),
        K=K,
        position=eye.copy(),
        yaw=yaw,
    )


def pose_from_c2w(
    c2w_flat: list,
    K:        torch.Tensor,
    device:   str,
) -> CameraPose:
    """
    c2w 4×4（行优先16值）→ CameraPose
    --c2w 参数对应此函数。
    """
    c2w = np.array(c2w_flat, dtype=np.float64).reshape(4, 4)
    view_mat = np.linalg.inv(c2w)
    position = c2w[:3, 3]
    forward  = -c2w[:3, 2]              # 相机看向 -Z
    yaw      = math.atan2(forward[0], forward[2])

    return CameraPose(
        view_matrix=torch.tensor(view_mat, dtype=torch.float32).unsqueeze(0).to(device),
        K=K,
        position=position.copy(),
        yaw=yaw,
    )


def poses_from_json(
    json_path: str,
    K:         torch.Tensor,
    device:    str,
) -> list:
    """
    批量 JSON → list[( name, CameraPose )]
    --cameras_json 参数对应此函数。

    JSON 格式（每项支持 look-at 或 c2w 两种写法）：
    [
      { "name": "frame_001",
        "eye": [0,1.5,3], "target": [0,0,0], "up": [0,1,0] },
      { "name": "frame_002",
        "c2w": [1,0,0,0, 0,1,0,1.5, 0,0,1,3, 0,0,0,1] }
    ]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    results = []
    for entry in entries:
        name = entry.get("name", f"frame_{len(results):04d}")
        if "c2w" in entry:
            pose = pose_from_c2w(entry["c2w"], K, device)
        else:
            pose = pose_from_lookat(
                np.array(entry["eye"],    dtype=np.float64),
                np.array(entry["target"], dtype=np.float64),
                np.array(entry.get("up", [0, 1, 0]), dtype=np.float64),
                K, device,
            )
        results.append((name, pose))

    print(f"[poses_from_json] 加载了 {len(results)} 个相机，来自 {json_path}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. 位姿更新（模型动作驱动）
# ══════════════════════════════════════════════════════════════════════════════

def update_camera_pose(
    current_pose: CameraPose,
    model_action:  tuple,           # (dx, dy, dz, dyaw)
    gaussians:     Gaussians = None,
    y_min:         float     = None,
    bbox:          tuple     = None,  # (min_xyz, max_xyz)
    collision_radius:  float = 0.15,
    collision_thresh:  int   = 50,
) -> CameraPose:
    """
    根据模型输出 (dx, dy, dz, dyaw) 更新相机位姿，
    并应用高度/边界框/碰撞约束。
    """
    dx, dy, dz, dyaw = model_action
    new_yaw = current_pose.yaw + dyaw

    cy, sy   = math.cos(current_pose.yaw), math.sin(current_pose.yaw)
    world_dx =  cy * dx - sy * dz
    world_dy =  dy
    world_dz =  sy * dx + cy * dz

    new_pos = current_pose.position + np.array([world_dx, world_dy, world_dz])

    # ── 约束 1：高度下限 ──────────────────────────────────────────────────────
    if y_min is not None:
        new_pos[1] = max(new_pos[1], y_min)

    # ── 约束 2：场景边界框 ────────────────────────────────────────────────────
    if bbox is not None:
        new_pos = np.clip(new_pos, bbox[0], bbox[1])

    # ── 约束 3：碰撞检测（穿墙保护）─────────────────────────────────────────
    if gaussians is not None:
        pos_t  = torch.tensor(new_pos, dtype=torch.float32,
                              device=gaussians.means.device)
        dist   = torch.norm(gaussians.means - pos_t, dim=-1)
        if (dist < collision_radius).sum().item() > collision_thresh:
            new_pos = current_pose.position   # 取消移动

    # ── 重建 view_matrix ──────────────────────────────────────────────────────
    ncy, nsy = math.cos(new_yaw), math.sin(new_yaw)
    forward  = np.array([-nsy, 0.0, -ncy])
    right    = np.array([ ncy, 0.0, -nsy])
    up_vec   = np.array([0.0,  1.0,  0.0])
    R        = np.stack([right, up_vec, -forward], axis=0)
    t        = -R @ new_pos
    view     = np.eye(4, dtype=np.float64)
    view[:3, :3] = R
    view[:3,  3] = t

    return CameraPose(
        view_matrix=torch.tensor(view, dtype=torch.float32).unsqueeze(0).to(
            current_pose.view_matrix.device),
        K=current_pose.K,
        position=new_pos.copy(),
        yaw=new_yaw,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. 渲染
# ══════════════════════════════════════════════════════════════════════════════

def render(
    gaussians:    Gaussians,
    camera_pose:  CameraPose,
    image_height: int = 512,
    image_width:  int = 512,
) -> torch.Tensor:
    """
    调用 gsplat rasterization 渲染一帧。
    返回 (H, W, 3) float32，值域 [0, 1]。
    """
    image, _alpha, _info = rasterization(
        means=gaussians.means,
        quats=gaussians.quats,
        scales=gaussians.scales,
        opacities=gaussians.opacities,
        colors=gaussians.colors,
        viewmats=camera_pose.view_matrix,   # (1, 4, 4)
        Ks=camera_pose.K,                   # (1, 3, 3)
        width=image_width,
        height=image_height,
        near_plane=0.01,
        far_plane=100.0,
        render_mode="RGB",
    )
    return image[0]   # (H, W, 3)


# ══════════════════════════════════════════════════════════════════════════════
# 6. 保存图像
# ══════════════════════════════════════════════════════════════════════════════

def save_image(tensor: torch.Tensor, path: str):
    """(H, W, 3) float32 → PNG"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = (tensor.detach().cpu().clamp(0, 1).numpy() * 255).astype("uint8")
    Image.fromarray(arr).save(str(out))
    print(f"[save_image] 已保存: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. 占位模型（替换为实际策略网络）
# ══════════════════════════════════════════════════════════════════════════════

def dummy_model(image: torch.Tensor) -> tuple:
    """返回 (dx, dy, dz, dyaw)。请替换为你的实际模型推理。"""
    return (0.0, 0.0, -0.1, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# 8. 命令行参数
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="3DGS 渲染器",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # PLY
    p.add_argument("--ply", required=True,
                   help="3DGS PLY 文件路径，例如 asset/point_cloud.ply")

    # 相机方式（三选一）
    cam = p.add_argument_group("相机输入（三选一）")
    cam.add_argument("--cameras_json",
                     help="批量相机外参 JSON 文件")
    cam.add_argument("--eye",    nargs=3, type=float,
                     metavar=("X", "Y", "Z"),
                     help="look-at: 相机位置")
    cam.add_argument("--target", nargs=3, type=float,
                     metavar=("X", "Y", "Z"),
                     help="look-at: 目标点")
    cam.add_argument("--up",     nargs=3, type=float,
                     default=[0, 1, 0],
                     metavar=("X", "Y", "Z"),
                     help="look-at: 上方向（默认 0 1 0）")
    cam.add_argument("--c2w",    nargs=16, type=float,
                     metavar="M",
                     help="camera-to-world 4×4 矩阵（行优先，16 个值）")

    # 输出
    p.add_argument("--output", default="output/frame.png",
                   help="单帧输出路径（.png）；批量模式下视为输出目录")

    # 渲染参数
    p.add_argument("--width",   type=int,   default=512)
    p.add_argument("--height",  type=int,   default=512)
    p.add_argument("--fov",     type=float, default=60.0,
                   help="水平视角（度），默认 60")

    # 动作循环（单视角模式下可选）
    p.add_argument("--steps",   type=int,   default=1,
                   help="渲染帧数（>1 时启用模型动作循环）")

    # 约束
    con = p.add_argument_group("位置约束")
    con.add_argument("--y_min",  type=float, default=None,
                     help="相机 Y 轴最低高度（防穿地板）")
    con.add_argument("--no_collision", action="store_true",
                     help="禁用碰撞检测（节省显存）")
    con.add_argument("--no_bbox",      action="store_true",
                     help="禁用场景边界框约束")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    W, H   = args.width, args.height

    print(f"使用设备: {DEVICE}")

    # ── 加载场景 ──────────────────────────────────────────────────────────────
    gaussians = load_ply(args.ply, device=DEVICE)
    mn, mx, center = scene_bounds(gaussians)
    print(f"场景范围  X: [{mn[0]:.2f}, {mx[0]:.2f}]  "
          f"Y: [{mn[1]:.2f}, {mx[1]:.2f}]  "
          f"Z: [{mn[2]:.2f}, {mx[2]:.2f}]")
    print(f"场景中心: {center}")

    # ── 构造内参 ──────────────────────────────────────────────────────────────
    K = make_K(args.fov, W, H, DEVICE)

    # ── 自动约束参数 ──────────────────────────────────────────────────────────
    y_min = args.y_min if args.y_min is not None else float(mn[1]) + 0.3
    bbox  = None if args.no_bbox else (mn, mx)

    # ══════════════════════════════════════════════════════════════════════════
    # 批量 JSON 模式
    # ══════════════════════════════════════════════════════════════════════════
    if args.cameras_json:
        out_dir = Path(args.output)
        poses   = poses_from_json(args.cameras_json, K, DEVICE)

        for name, pose in poses:
            with torch.no_grad():
                img = render(gaussians, pose, H, W)
            save_image(img, str(out_dir / f"{name}.png"))

        print(f"\n批量渲染完成，共 {len(poses)} 帧，输出目录: {out_dir}")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # 单视角 / 动作循环模式
    # ══════════════════════════════════════════════════════════════════════════

    # ── 构造初始位姿 ──────────────────────────────────────────────────────────
    if args.c2w:
        current_pose = pose_from_c2w(args.c2w, K, DEVICE)
        print(f"[c2w]    相机位置: {current_pose.position}")

    elif args.eye and args.target:
        current_pose = pose_from_lookat(
            np.array(args.eye,    dtype=np.float64),
            np.array(args.target, dtype=np.float64),
            np.array(args.up,     dtype=np.float64),
            K, DEVICE,
        )
        print(f"[look-at] eye={args.eye}  target={args.target}")

    else:
        # 未指定视角：自动放在场景中心前方
        default_eye = center + np.array([0.0, 0.0, np.linalg.norm(mx - mn) * 0.6])
        current_pose = pose_from_lookat(
            eye=default_eye,
            target=center,
            up=np.array([0.0, 1.0, 0.0]),
            K=K,
            device=DEVICE,
        )
        print(f"[auto]   未指定视角，自动设置 eye={default_eye.round(2)}")

    # ── 渲染循环 ──────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    single_frame = args.steps == 1

    for step in range(args.steps):
        print(f"\n── Step {step} ──────────────────────────────────")

        with torch.no_grad():
            img = render(gaussians, current_pose, H, W)

        print(f"  图像 shape={img.shape}  "
              f"值域=[{img.min():.3f}, {img.max():.3f}]")

        # 输出路径：单帧直接用 --output，多帧自动编号
        if single_frame:
            out_file = output_path
        else:
            stem = output_path.stem
            out_file = output_path.parent / f"{stem}_{step:04d}.png"

        save_image(img, str(out_file))

        if step < args.steps - 1:
            action = dummy_model(img)
            print(f"  动作: dx={action[0]:.3f} dy={action[1]:.3f} "
                  f"dz={action[2]:.3f} dyaw={action[3]:.3f}")

            current_pose = update_camera_pose(
                current_pose, action,
                gaussians=None if args.no_collision else gaussians,
                y_min=y_min,
                bbox=bbox,
            )
            print(f"  新位置: {current_pose.position.round(3)}  "
                  f"yaw={current_pose.yaw:.3f} rad")

    print("\n渲染完成。")


if __name__ == "__main__":
    main()