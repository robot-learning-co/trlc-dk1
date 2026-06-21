"""DK1 kinematics shim — radian I/O over LeRobot's placo-backed RobotKinematics."""

import numpy as np

from lerobot.model.kinematics import RobotKinematics


class DK1RobotKinematics(RobotKinematics):
    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str,
        joint_names: list[str] | None = None,
        *,
        kinetic_reg: float | None = 1e-4,
        solver_dt: float = 1 / 30,
        converge_ik: bool = False,
        ik_max_iters: int = 20,
        ik_pos_tol: float = 1e-3,
        ik_ori_tol: float = 1e-2,
    ):
        super().__init__(urdf_path, target_frame_name, joint_names)
        self.solver.dt = solver_dt
        if kinetic_reg is not None:
            self.solver.add_kinetic_energy_regularization_task(kinetic_reg)
        self.converge_ik = converge_ik
        self.ik_max_iters = ik_max_iters
        self.ik_pos_tol = ik_pos_tol
        self.ik_ori_tol = ik_ori_tol

    def forward_kinematics(self, joint_pos_rad: np.ndarray) -> np.ndarray:
        q = np.asarray(joint_pos_rad, dtype=float)
        n = len(self.joint_names)
        if q.size < n:
            raise ValueError(f"forward_kinematics: need at least {n} joint angles, got {q.size}")
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

        max_iters = self.ik_max_iters if self.converge_ik else 1
        for _ in range(max_iters):
            self.solver.solve(True)
            self.robot.update_kinematics()
            if not self.converge_ik:
                break
            t_now = self.robot.get_T_world_frame(self.target_frame_name)
            pos_err = float(np.linalg.norm(t_now[:3, 3] - desired_ee_pose[:3, 3]))
            r_err = t_now[:3, :3].T @ desired_ee_pose[:3, :3]
            ori_err = float(np.arccos(np.clip((np.trace(r_err) - 1.0) / 2.0, -1.0, 1.0)))
            if pos_err <= self.ik_pos_tol and ori_err <= self.ik_ori_tol:
                break

        q_out = np.array(
            [self.robot.get_joint(name) for name in self.joint_names], dtype=float
        )

        if current.size > n:
            result = current.copy()
            result[:n] = q_out
            return result
        return q_out
