import numpy as np
import torch
import uuid
from pathlib import Path

from api.render_3dgs import (
    load_ply,
    make_K,
    pose_from_lookat,
    render,
    save_image
)


class Renderer3DGS:

    def __init__(
        self,
        ply_path,
        width=512,
        height=512,
        fov=60.0,
        device=None
    ):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print("Renderer device:", self.device)

        self.width = width
        self.height = height

        # 只加载一次 PLY
        self.gaussians = load_ply(ply_path, device=self.device)

        # 相机内参
        self.K = make_K(fov, width, height, self.device)

        # 输出目录
        self.out_dir = Path("tmp_renders")
        self.out_dir.mkdir(exist_ok=True)

    def render(self, camera, target):

        eye = np.array(camera, dtype=np.float64)
        tgt = np.array(target, dtype=np.float64)

        # 构造相机位姿
        pose = pose_from_lookat(
            eye,
            tgt,
            np.array([0, 1, 0]),
            self.K,
            self.device
        )

        with torch.no_grad():

            img = render(
                self.gaussians,
                pose,
                self.height,
                self.width
            )

        out_path = self.out_dir / f"{uuid.uuid4().hex}.png"

        save_image(img, str(out_path))

        return str(out_path)