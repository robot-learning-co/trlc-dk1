"""DK1 kinematics shim — radian I/O over LeRobot's placo-backed RobotKinematics.

LeRobot's `RobotKinematics` assumes joint angles are in **degrees** on both input
and output. DK1's DM-series follower motors and the Dynamixel leader both report
positions in **radians**, so this subclass overrides FK and IK to skip the
deg↔rad conversions. Defaults otherwise match upstream (see `inverse_kinematics`).
"""

from __future__ import annotations

import numpy as np

from lerobot.model.kinematics import RobotKinematics


class DK1RobotKinematics(RobotKinematics):
    """placo-backed FK/IK for DK1 with radian I/O instead of degrees."""

    def forward_kinematics(self, joint_pos_rad: np.ndarray) -> np.ndarray:
        q = np.asarray(joint_pos_rad, dtype=float)
        n = len(self.joint_names)
        if q.size < n:
            raise ValueError(
                f"forward_kinematics: need at least {n} joint angles, got {q.size}"
            )
        q = q[:n]
        for name, val in zip(self.joint_names, q):
            self.robot.set_joint(name, float(val))
        self.robot.update_kinematics()
        return self.robot.get_T_world_frame(self.target_frame_name)

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> np.ndarray:
        current = np.asarray(current_joint_pos, dtype=float)
        n = len(self.joint_names)
        if current.size < n:
            raise ValueError(
                f"inverse_kinematics: current_joint_pos needs at least {n} entries, got {current.size}"
            )
        q_seed = current[:n]
        for name, val in zip(self.joint_names, q_seed):
            self.robot.set_joint(name, float(val))

        self.tip_frame.T_world_frame = desired_ee_pose
        self.tip_frame.configure(
            self.target_frame_name, "soft", position_weight, orientation_weight
        )
        self.solver.solve(True)
        self.robot.update_kinematics()

        q_out = np.array(
            [self.robot.get_joint(name) for name in self.joint_names], dtype=float
        )

        # Preserve any trailing entries (e.g. gripper) the caller passed in.
        if current.size > len(self.joint_names):
            result = current.copy()
            result[: len(self.joint_names)] = q_out
            return result
        return q_out
