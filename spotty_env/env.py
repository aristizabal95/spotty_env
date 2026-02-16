"""
Spotty Environment for Genesis Physics Simulator

Based on Go2Env from Genesis examples:
https://github.com/Genesis-Embodied-AI/Genesis/blob/main/examples/locomotion/go2_env.py

This environment provides a Gymnasium-compatible interface for controlling the Spotty
robot, with normalized action space [0, 1] for each of the 12 revolute joints.
"""

import pathlib
import numpy as np

import gymnasium as gym
from gymnasium import spaces

import genesis as gs
gs.init()


class SpottyEnv(gym.Env):
    """
    Environment for controlling the Spotty robot in Genesis simulator.

    Actions are normalized to [0, 1] range, where:
    - 0.0 maps to the minimum joint angle (lower limit)
    - 1.0 maps to the maximum joint angle (upper limit)

    Rendering follows Gymnasium conventions:
    - render_mode=None (default): headless, no display or camera.
    - render_mode="human": interactive viewer in a separate window.
    - render_mode="rgb_array": headless; render() returns an RGB array (e.g. for matplotlib).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        dt=0.01,
        kp_gain=16000.0,
        kv_gain=0.0,
        render_mode=None,
        fixed_base=False,
        joint_reverse=None,
        num_scene_steps_per_env_step=1,
        n_envs=1,
        env_spacing=(1.0, 1.0),
        render_res=(640, 480),
    ):
        """
        Initialize the Spotty environment.

        Args:
            dt: Simulation time step in seconds (default: 0.01 = 100 Hz)
            kp_gain: Proportional gain for PD controller (default: 8000.0)
            kv_gain: Derivative gain for PD controller (default: 200.0)
            render_mode: None (headless), "human" (viewer window), or "rgb_array"
                        (headless; render() returns RGB array for matplotlib).
            fixed_base: Whether to fix the base link to the world (default: False)
            joint_reverse: Vector of size 12 with 1 or 0 indicating if a joint
                          needs to be reversed. If None, defaults to all zeros.
            num_scene_steps_per_env_step: Number of physics scene steps per env step.
            n_envs: Number of parallel environments to simulate (default: 1).
            env_spacing: (x, y) spacing between environment origins (default: (1.0, 1.0)).
            render_res: (width, height) for rgb_array rendering (default: (640, 480)).
        """
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}, got {render_mode!r}"
            )
        # gym.Env has no __init__, so we set render_mode ourselves
        self.render_mode = render_mode
        self._render_res = render_res

        self.dt = dt
        self.kp_gain = kp_gain
        self.kv_gain = kv_gain
        self.fixed_base = fixed_base
        self.num_scene_steps_per_env_step = num_scene_steps_per_env_step
        
        # Get URDF path
        script_dir = pathlib.Path(__file__).parent
        self.urdf_path = script_dir / "robot" / "spotty.urdf"
        
        if not self.urdf_path.exists():
            raise FileNotFoundError(
                f"URDF file not found: {self.urdf_path}\n"
                "Please ensure the URDF file exists."
            )
        
        # Viewer only for "human" mode; headless otherwise
        show_viewer = self.render_mode == "human"
        self._camera_pos = (3.5, 0.0, 2.5)
        self._camera_lookat = (0.0, 0.0, 0.5)
        self._camera_fov = 40

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                gravity=(0, 0, -9.81),
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=True,
                world_frame_size=1.0,
                show_link_frame=False,
                show_cameras=False,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=self._camera_pos,
                camera_lookat=self._camera_lookat,
                camera_fov=self._camera_fov,
                max_FPS=self.metadata["render_fps"],
            ),
            show_viewer=show_viewer,
        )

        # Add ground plane
        self.scene.add_entity(
            morph=gs.morphs.Plane(pos=(0.0, 0.0, 0.0))
        )

        # Add robot
        self.robot = self.scene.add_entity(
            morph=gs.morphs.URDF(
                file=str(self.urdf_path),
                pos=(0.0, 0.0, -0.25),
                euler=(0.0, 0.0, 0.0),
                scale=1.0,
                fixed=self.fixed_base,
            )
        )

        # Headless camera for rgb_array; must be added before build()
        self._render_camera = None
        if self.render_mode == "rgb_array":
            self._render_camera = self.scene.add_camera(
                res=self._render_res,
                pos=self._camera_pos,
                lookat=self._camera_lookat,
                fov=self._camera_fov,
                GUI=False,
            )

        # Build scene
        self.scene.build(n_envs=n_envs, env_spacing=env_spacing)
        self.n_envs = self.scene.n_envs

        # Identify controllable joints and get their limits
        self._setup_joints()
        
        # Set up joint reversal vector
        if joint_reverse is None:
            joint_reverse = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        joint_reverse = np.array(joint_reverse, dtype=np.float32)
        if joint_reverse.shape != (self.num_actions,):
            raise ValueError(
                f"joint_reverse must have shape ({self.num_actions},), "
                f"got {joint_reverse.shape}"
            )
        # Ensure values are 0 or 1
        self.joint_reverse = np.clip(joint_reverse, 0.0, 1.0)
        
        # Set up PD control
        self._setup_control()

        # Gymnasium spaces (for one environment)
        self._build_spaces()

        # Get initial DOF positions and convert to numpy array (handles CUDA tensors)
        initial_pos = self.robot.get_dofs_position()
        self.initial_dofs_position = initial_pos.cpu().numpy()

        print(
            f"SpottyEnv initialized with {self.num_actions} controllable joints "
            f"and {self.n_envs} parallel envs"
        )
        print(f"Joint reversal vector: {self.joint_reverse}")
    
    def _setup_joints(self):
        """Identify revolute joints and extract their limits."""
        joint_names = []
        motors_dof_idx = []
        joint_lower_limits = []
        joint_upper_limits = []
        
        for joint in self.robot.joints:
            if joint.type == gs.JOINT_TYPE.REVOLUTE:
                joint_names.append(joint.name)
                motors_dof_idx.append(joint.dofs_idx_local[0])
                
                # Get joint limits
                if hasattr(joint, 'limit'):
                    lower = joint.limit.lower if hasattr(joint.limit, 'lower') else -np.pi
                    upper = joint.limit.upper if hasattr(joint.limit, 'upper') else np.pi
                else:
                    lower = -np.pi
                    upper = np.pi
                
                joint_lower_limits.append(lower)
                joint_upper_limits.append(upper)
        
        self.joint_names = joint_names
        self.motors_dof_idx = motors_dof_idx
        self.joint_lower_limits = np.array(joint_lower_limits)
        self.joint_upper_limits = np.array(joint_upper_limits)
        self.num_actions = len(motors_dof_idx)
        
        print(f"Found {self.num_actions} revolute joints:")
        for name, lower, upper in zip(joint_names, joint_lower_limits, joint_upper_limits):
            print(f"  {name}: [{lower:.4f}, {upper:.4f}] rad")
    
    def _setup_control(self):
        """Set up PD control gains."""
        kp_values = np.array([self.kp_gain] * self.num_actions)
        kv_values = np.array([self.kv_gain] * self.num_actions)

        self.robot.set_dofs_kp(kp=kp_values, dofs_idx_local=self.motors_dof_idx)
        self.robot.set_dofs_kv(kv=kv_values, dofs_idx_local=self.motors_dof_idx)

    def _build_spaces(self):
        """Build Gymnasium observation_space and action_space (single env)."""
        low = np.array(self.joint_lower_limits, dtype=np.float32)
        high = np.array(self.joint_upper_limits, dtype=np.float32)
        # Joint velocities: use a large bound (rad/s)
        vel_bound = 50.0
        # Base position: large bound (meters)
        pos_bound = 100.0
        # Quaternion: each component in [-1, 1]
        self.observation_space = spaces.Dict(
            {
                "joint_positions": spaces.Box(low=low, high=high, dtype=np.float32),
                "normalized_positions": spaces.Box(
                    low=0.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
                ),
                "joint_velocities": spaces.Box(
                    low=-vel_bound,
                    high=vel_bound,
                    shape=(self.num_actions,),
                    dtype=np.float32,
                ),
                "base_position": spaces.Box(
                    low=-pos_bound, high=pos_bound, shape=(3,), dtype=np.float32
                ),
                "base_quaternion": spaces.Box(
                    low=-1.0, high=1.0, shape=(4,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

    def normalized_to_joint_angles(self, actions):
        """
        Convert normalized actions [0, 1] to joint angles in radians.

        Supports both single-env and batched actions for parallel simulation.

        Args:
            actions: Array of shape (num_actions,) or (n_envs, num_actions)
                    with values in [0, 1]. If 1D, the same action is applied
                    to all envs.

        Returns:
            Array of joint angles in radians, shape (n_envs, num_actions)
        """
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            if actions.shape != (self.num_actions,):
                raise ValueError(
                    f"Actions must have shape ({self.num_actions},) or "
                    f"({self.n_envs}, {self.num_actions}), got {actions.shape}"
                )
            actions = np.broadcast_to(
                actions[np.newaxis, :], (self.n_envs, self.num_actions)
            ).copy()
        elif actions.ndim == 2:
            if actions.shape != (self.n_envs, self.num_actions):
                raise ValueError(
                    f"Batched actions must have shape ({self.n_envs}, {self.num_actions}), "
                    f"got {actions.shape}"
                )
        else:
            raise ValueError(
                f"Actions must be 1D or 2D, got ndim={actions.ndim}"
            )

        # Clamp actions to [0, 1]
        actions = np.clip(actions, 0.0, 1.0)
        # Reverse actions based on joint_reverse vector (broadcasts over envs)
        actions = np.abs(actions - self.joint_reverse)

        # Linear interpolation: 0 -> lower_limit, 1 -> upper_limit
        joint_angles = (
            self.joint_lower_limits
            + actions * (self.joint_upper_limits - self.joint_lower_limits)
        )
        return joint_angles
    
    def joint_angles_to_normalized(self, joint_angles):
        """
        Convert joint angles in radians to normalized actions [0, 1].

        Accepts both single-env and batched joint angles (from get_dofs_position).

        Args:
            joint_angles: Array of shape (num_actions,) or (n_envs, num_actions)

        Returns:
            Array of normalized actions in [0, 1], same shape as input
        """
        joint_angles = np.asarray(joint_angles, dtype=np.float32)
        # Inverse linear interpolation (limits broadcast over env dim if 2D)
        actions = (
            (joint_angles - self.joint_lower_limits)
            / (self.joint_upper_limits - self.joint_lower_limits)
        )
        actions = np.clip(actions, 0.0, 1.0)
        actions = np.abs(actions - self.joint_reverse)
        return actions
    
    def step(self, action):
        """
        Step the simulation with normalized actions.

        Args:
            action: Array of shape (num_actions,) or (n_envs, num_actions)
                    in [0, 1]. If 1D, the same action is applied to all envs.

        Returns:
            observation: Dict of batched state; each value has shape (n_envs, ...).
            reward: float (or array of shape (n_envs,) when n_envs > 1, for vector use).
            terminated: bool (or (n_envs,) when n_envs > 1).
            truncated: bool (or (n_envs,) when n_envs > 1).
            info: dict (may contain batched extras when n_envs > 1).
        """
        joint_angles = self.normalized_to_joint_angles(action)
        joint_angles = np.ascontiguousarray(joint_angles, dtype=np.float32)

        for _ in range(self.num_scene_steps_per_env_step):
            self.robot.control_dofs_position(
                joint_angles,
                self.motors_dof_idx,
            )
            self.scene.step()

        obs = self.get_observations()
        reward = np.zeros(self.n_envs, dtype=np.float32)
        terminated = np.zeros(self.n_envs, dtype=bool)
        truncated = np.zeros(self.n_envs, dtype=bool)
        info = {}

        if self.n_envs == 1:
            obs_single = {k: np.squeeze(v, axis=0) for k, v in obs.items()}
            return obs_single, float(reward[0]), bool(terminated[0]), bool(truncated[0]), info
        return obs, reward, terminated, truncated, info
    
    def get_observations(self):
        """
        Get current observations from the robot (batched over n_envs).

        Returns:
            dict: Each value has shape (n_envs, ...).
        """
        obs = {}

        if hasattr(self.robot, 'get_dofs_position'):
            joint_pos = self.robot.get_dofs_position(self.motors_dof_idx)
            obs['joint_positions'] = joint_pos.cpu().numpy()
            obs['normalized_positions'] = self.joint_angles_to_normalized(
                obs['joint_positions']
            )
        else:
            obs['joint_positions'] = np.zeros(
                (self.n_envs, self.num_actions), dtype=np.float32
            )
            obs['normalized_positions'] = np.zeros(
                (self.n_envs, self.num_actions), dtype=np.float32
            )

        if hasattr(self.robot, 'get_dofs_velocity'):
            joint_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)
            obs['joint_velocities'] = joint_vel.cpu().numpy()
        else:
            obs['joint_velocities'] = np.zeros(
                (self.n_envs, self.num_actions), dtype=np.float32
            )

        if hasattr(self.robot, 'get_pos'):
            base_pos = self.robot.get_pos()
            obs['base_position'] = base_pos.cpu().numpy()
        else:
            obs['base_position'] = np.zeros(
                (self.n_envs, 3), dtype=np.float32
            )

        if hasattr(self.robot, 'get_quat'):
            base_quat = self.robot.get_quat()
            obs['base_quaternion'] = base_quat.cpu().numpy()
        else:
            obs['base_quaternion'] = np.zeros(
                (self.n_envs, 4), dtype=np.float32
            )
            obs['base_quaternion'][:, 0] = 1.0

        return obs
    
    def reset(self, seed=None, options=None):
        """
        Reset the simulation to initial state.

        Args:
            seed: Optional RNG seed (passed to Gymnasium).
            options: Optional dict (unused).

        Returns:
            observation: Dict of (batched) state; each value shape (n_envs, ...).
            info: dict.
        """
        super().reset(seed=seed)
        self.robot.set_dofs_position(self.initial_dofs_position)
        zero_velocities = np.zeros_like(self.initial_dofs_position)
        self.robot.set_dofs_velocity(zero_velocities)
        self.scene.step()
        obs = self.get_observations()
        info = {}
        if self.n_envs == 1:
            obs_single = {k: np.squeeze(v, axis=0) for k, v in obs.items()}
            return obs_single, info
        return obs, info
    
    def render(self):
        """
        Render the current state.

        Returns:
            None if render_mode is None or "human".
            np.ndarray of shape (height, width, 3) uint8 RGB if render_mode is "rgb_array"
            (suitable for matplotlib.imshow or saving).
        """
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            # Viewer updates automatically on scene.step(); nothing to return
            return None
        if self.render_mode == "rgb_array":
            out = self._render_camera.render()
            rgb = out[0] if isinstance(out, (list, tuple)) else out
            if hasattr(rgb, "cpu"):
                rgb = rgb.cpu().numpy()
            rgb = np.asarray(rgb, dtype=np.uint8)
            # Ensure (H, W, 3) for matplotlib (batch dim if present)
            if rgb.ndim == 4:
                rgb = rgb[0]
            return rgb
        return None

    def close(self):
        """Close the environment and cleanup resources."""
        # Genesis will handle cleanup when scene is garbage collected
        pass


def main():
    """Example usage of SpottyEnv with parallel simulation and rendering."""
    import sys

    # Default: headless. Use --human for viewer, --rgb for rgb_array + matplotlib
    render_mode = None
    if "--human" in sys.argv:
        render_mode = "human"
    elif "--rgb" in sys.argv:
        render_mode = "rgb_array"

    env = SpottyEnv(
        render_mode=render_mode,
        num_scene_steps_per_env_step=10,
        n_envs=10,
    )

    print("\n=== Example: Moving joints with normalized actions (batched) ===")
    print(f"Render mode: {env.render_mode!r}")

    obs, info = env.reset()
    print(f"Initial normalized positions shape: {obs['normalized_positions'].shape}")
    print(f"Initial normalized positions (env 0): {obs['normalized_positions'][0]}")

    # Same action for all envs: pass (num_actions,) and it is broadcast
    print("\nMoving all joints to maximum (1.0) in all envs...")
    for _ in range(100):
        actions = np.ones(env.num_actions)
        obs, reward, terminated, truncated, info = env.step(actions)
        if env.render_mode == "human":
            env.render()

    # Batched actions: (n_envs, num_actions)
    print("Moving all joints to minimum (0.0) in all envs...")
    for _ in range(100):
        actions = np.zeros((env.n_envs, env.num_actions))
        obs, reward, terminated, truncated, info = env.step(actions)
        if env.render_mode == "human":
            env.render()

    # Random walk
    print("Performing random walk (different action per env)...")
    for step in range(200):
        actions = env.action_space.sample()
        if env.n_envs > 1:
            actions = np.broadcast_to(
                actions[np.newaxis, :], (env.n_envs, env.num_actions)
            ).copy()
        obs, reward, terminated, truncated, info = env.step(actions)
        if env.render_mode == "human":
            env.render()
        if step % 50 == 0:
            print(
                f"Step {step}: normalized positions (env 0) = "
                f"{obs['normalized_positions'][0]}"
            )

    if env.render_mode == "rgb_array":
        frame = env.render()
        if frame is not None:
            try:
                import matplotlib.pyplot as plt
                plt.imshow(frame)
                plt.axis("off")
                plt.title("SpottyEnv (rgb_array)")
                plt.tight_layout()
                plt.savefig("spotty_env_frame.png", dpi=100, bbox_inches="tight")
                print("\nSaved last frame to spotty_env_frame.png")
            except ImportError:
                print("\nmatplotlib not installed; skipping save of rgb_array frame")

    env.close()
    print("\nExample completed!")


if __name__ == "__main__":
    main()

