"""
Spotty Environment for Genesis Physics Simulator

Based on Go2Env from Genesis examples:
https://github.com/Genesis-Embodied-AI/Genesis/blob/main/examples/locomotion/go2_env.py

This environment provides a class-based interface for controlling the Spotty robot,
with normalized action space [0, 1] for each of the 12 revolute joints.
"""

import pathlib
import numpy as np

try:
    import genesis as gs
    # Try to initialize Genesis if init function exists
    if hasattr(gs, 'init'):
        gs.init()
except ImportError as exc:
    raise ImportError(
        "Genesis physics simulator not found. "
        "Please install the correct Genesis physics simulator package. "
        "Check: https://genesis-world.readthedocs.io/en/latest/"
        "user_guide/overview/installation.html"
    ) from exc


class SpottyEnv:
    """
    Environment for controlling the Spotty robot in Genesis simulator.
    
    Actions are normalized to [0, 1] range, where:
    - 0.0 maps to the minimum joint angle (lower limit)
    - 1.0 maps to the maximum joint angle (upper limit)
    """
    
    def __init__(self, dt=0.01, kp_gain=16000.0, kv_gain=0.0, 
                 show_viewer=True, fixed_base=False, joint_reverse=None,
                 num_scene_steps_per_env_step=1):
        """
        Initialize the Spotty environment.
        
        Args:
            dt: Simulation time step in seconds (default: 0.01 = 100 Hz)
            kp_gain: Proportional gain for PD controller (default: 8000.0)
            kv_gain: Derivative gain for PD controller (default: 200.0,
                    approximately 1/40 of kp for balanced response)
            show_viewer: Whether to show the visualization window (default: True)
            fixed_base: Whether to fix the base link to the world (default: False)
            joint_reverse: Vector of size 12 with 1 or 0 indicating if a joint
                          needs to be reversed. 1 means reverse (1.0 - action),
                          0 means no reversal. If None, defaults to all zeros.
            num_scene_steps_per_env_step: Number of physics scene steps to
                                         execute per environment step (default: 1).
                                         This simulates real-life scenarios where
                                         each env step happens over a few
                                         milliseconds. Higher values provide
                                         finer physics simulation.
        """
        self.dt = dt
        self.kp_gain = kp_gain
        self.kv_gain = kv_gain
        self.show_viewer = show_viewer
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
        
        # Create scene
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
                camera_pos=(3.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=100,
            ),
            show_viewer=self.show_viewer,
        )
        
        # Add ground plane
        self.scene.add_entity(
            morph=gs.morphs.Plane(
                pos=(0.0, 0.0, 0.0)
            )
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
        
        # Build scene
        self.scene.build()
        
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

        # Get initial DOF positions and convert to numpy array (handles CUDA tensors)
        initial_pos = self.robot.get_dofs_position()
        self.initial_dofs_position = initial_pos.cpu().numpy()
        
        print(f"SpottyEnv initialized with {self.num_actions} controllable joints")
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
    
    def normalized_to_joint_angles(self, actions):
        """
        Convert normalized actions [0, 1] to joint angles in radians.
        
        Args:
            actions: Array of shape (num_actions,) with values in [0, 1]
                    or list of 12 values between 0 and 1
        
        Returns:
            Array of joint angles in radians
        """
        actions = np.array(actions, dtype=np.float32)
        
        if actions.shape != (self.num_actions,):
            raise ValueError(
                f"Actions must have shape ({self.num_actions},), "
                f"got {actions.shape}"
            )
        
        # Clamp actions to [0, 1]
        actions = np.clip(actions, 0.0, 1.0)
        # Reverse actions based on joint_reverse vector
        actions = abs(actions - self.joint_reverse)
        
        # Linear interpolation: 0 -> lower_limit, 1 -> upper_limit
        joint_angles = (
            self.joint_lower_limits + 
            actions * (self.joint_upper_limits - self.joint_lower_limits)
        )
        
        return joint_angles
    
    def joint_angles_to_normalized(self, joint_angles):
        """
        Convert joint angles in radians to normalized actions [0, 1].
        
        Args:
            joint_angles: Array of joint angles in radians
        
        Returns:
            Array of normalized actions in [0, 1]
        """
        joint_angles = np.array(joint_angles, dtype=np.float32)
        
        # Inverse linear interpolation
        actions = (
            (joint_angles - self.joint_lower_limits) / 
            (self.joint_upper_limits - self.joint_lower_limits)
        )
        
        # Clamp to [0, 1]
        actions = np.clip(actions, 0.0, 1.0)
        actions = abs(actions - self.joint_reverse)
        
        return actions
    
    def step(self, actions):
        """
        Step the simulation with normalized actions.
        
        Args:
            actions: Array or list of 12 values in [0, 1] representing
                    normalized joint positions
        
        Returns:
            dict: Dictionary containing observation information
        """
        # Convert normalized actions to joint angles
        joint_angles = self.normalized_to_joint_angles(actions)
        
        # Step simulation multiple times to simulate fine-grained physics
        # Apply control at each physics step for faster response
        for _ in range(self.num_scene_steps_per_env_step):
            # Apply control at each physics step - this allows the PD controller
            # to update more frequently and respond faster
            self.robot.control_dofs_position(
                joint_angles,
                self.motors_dof_idx
            )
            self.scene.step()
        
        # Get current state
        obs = self.get_observations()
        
        return obs
    
    def get_observations(self):
        """
        Get current observations from the robot.
        
        Returns:
            dict: Dictionary containing current observations
        """
        obs = {}
        
        # Get current joint positions and velocities
        if hasattr(self.robot, 'get_dofs_position'):
            joint_pos = self.robot.get_dofs_position(self.motors_dof_idx)
            obs['joint_positions'] = joint_pos.cpu().numpy()
            obs['normalized_positions'] = self.joint_angles_to_normalized(
                obs['joint_positions']
            )
        
        if hasattr(self.robot, 'get_dofs_velocity'):
            joint_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)
            obs['joint_velocities'] = joint_vel.cpu().numpy()
        else:
            obs['joint_velocities'] = np.zeros(self.num_actions, dtype=np.float32)
        
        # Get base pose if available
        if hasattr(self.robot, 'get_pos'):
            base_pos = self.robot.get_pos()
            obs['base_position'] = base_pos.cpu().numpy()
        else:
            obs['base_position'] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        if hasattr(self.robot, 'get_quat'):
            base_quat = self.robot.get_quat()
            obs['base_quaternion'] = base_quat.cpu().numpy()
        else:
            obs['base_quaternion'] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        return obs
    
    def reset(self):
        """
        Reset the simulation to initial state.
        
        Args:
            initial_actions: Optional initial normalized actions [0, 1].
                           If None, uses middle positions (0.5 for all joints)
        
        Returns:
            dict: Initial observations
        """
        # Reset robot to initial pose (initial_dofs_position is already numpy array)
        self.robot.set_dofs_position(self.initial_dofs_position)
        
        # Reset velocities if possible (initial_dofs_position is numpy, so zeros_like works)
        zero_velocities = np.zeros_like(self.initial_dofs_position)
        self.robot.set_dofs_velocity(zero_velocities)
        
        self.scene.step()
        
        return self.get_observations()
    
    def close(self):
        """Close the environment and cleanup resources."""
        # Genesis will handle cleanup when scene is garbage collected
        pass


def main():
    """Example usage of SpottyEnv."""
    # Create environment
    env = SpottyEnv(show_viewer=True, num_scene_steps_per_env_step=10)
    
    print("\n=== Example: Moving joints with normalized actions ===")
    
    # Reset to default (middle positions)
    obs = env.reset()
    print(f"Initial normalized positions: {obs['normalized_positions']}")
    
    # Example: Move all joints to maximum
    print("\nMoving all joints to maximum (1.0)...")
    for step in range(100):
        actions = np.ones(env.num_actions)  # All joints at maximum
        obs = env.step(actions)
    
    # Example: Move all joints to minimum
    print("Moving all joints to minimum (0.0)...")
    for step in range(100):
        actions = np.zeros(env.num_actions)  # All joints at minimum
        obs = env.step(actions)
    
    # Example: Random walk
    print("Performing random walk...")
    np.random.seed(42)
    while True:
        # Random actions in [0, 1]
        actions = np.random.rand(env.num_actions)
        obs = env.step(actions)
        
        if step % 50 == 0:
            print(f"Step {step}: normalized positions = {obs['normalized_positions']}")
    
    print("\nExample completed!")


if __name__ == "__main__":
    main()

