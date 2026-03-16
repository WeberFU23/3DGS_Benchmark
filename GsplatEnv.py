import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch


class GSplatEnv(gym.Env):
    def __init__(self, ply_path, config):
        super(GSplatEnv, self).__init__()
        # 1. 初始化你的渲染器（从你的 render_3dgs.py 加载）
        self.gaussians = load_ply(ply_path, device="cuda")
        self.K = make_K(config.fov, config.width, config.height, "cuda")

        # 2. 定义动作空间 (Action Space)
        # 假设模型输出是 [dx, dy, dz, dyaw]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # 3. 定义观察空间 (Observation Space)
        # 即渲染出来的 RGB 图片数据结构
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(config.height, config.width, 3),
                                            dtype=np.uint8)

        self.current_pose = None

    def reset(self, seed=None, options=None):
        # 重置环境到初始位姿（可能是场景中心或特定起点）
        super().reset(seed=seed)
        self.current_pose = self.get_initial_pose()
        obs = self._render_current_view()
        return obs, {}

    def step(self, action):
        # 1. 应用你代码里的 update_camera_pose
        self.current_pose = update_camera_pose(
            self.current_pose, action,
            gaussians=self.gaussians,
            y_min=self.y_min, bbox=self.bbox
        )

        # 2. 渲染新视角
        obs = self._render_current_view()

        # 3. 计算奖励 (在 Benchmark 中，通常步数越多奖励越低，鼓励高效感知)
        reward = -0.1
        terminated = False  # 是否到达终点（例如模型决定提交答案）

        return obs, reward, terminated, False, {}

    def _render_current_view(self):
        # 调用你代码里的 render 函数
        with torch.no_grad():
            img_tensor = render(self.gaussians, self.current_pose)
        # 转为 numpy uint8 格式供模型/OpenCV使用
        return (img_tensor.cpu().numpy() * 255).astype(np.uint8)