"""
DK1Kinematics — MuJoCo-backed forward/inverse kinematics and Jacobian for the
TRLC-DK1 arm.

This is a *radians-native* drop-in for LeRobot's ``RobotKinematics`` interface
(``forward_kinematics`` / ``inverse_kinematics``), so it can be handed directly to
LeRobot's end-effector processor steps. Unlike LeRobot's placo-based
``RobotKinematics`` (which works in degrees), everything here is in radians —
matching the rest of the DK1 stack (``DK1Robot``, the follower observations/
actions, and the leader).

It reuses the same URDF/MuJoCo model used for gravity compensation
(see ``gravity_comp.py``), so the FK that reports the current EE pose and the
model that computes gravity/limits can never disagree. ``jacobian()`` is exposed
for a future task-space (Cartesian impedance) controller.

Kept free of any LeRobot import so it can live in the standalone control stack.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

import numpy as np

from .config import DK1RobotConfig, _DEFAULT_URDF
from .gravity_comp import _urdf_strip_meshes

logger = logging.getLogger(__name__)


def _rotation_matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to an axis-angle rotation vector (radians).

    Small-numpy implementation (avoids a scipy dependency in the control stack).
    """
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3)
    if np.pi - theta < 1e-6:
        # Near 180 deg: axis from the symmetric part (diagonal of R + I).
        axis = np.sqrt(np.clip((np.diag(R) + 1.0) / 2.0, 0.0, None))
        # Fix signs using off-diagonal terms.
        axis[0] = abs(axis[0])
        if axis[1] != 0:
            axis[1] = np.copysign(axis[1], R[0, 1])
        if axis[2] != 0:
            axis[2] = np.copysign(axis[2], R[0, 2])
        return axis / np.linalg.norm(axis) * theta
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2.0 * np.sin(theta)) * theta


class DK1Kinematics:
    """
    Forward/inverse kinematics + Jacobian for the DK1 arm, in radians.

    Args:
        model_path: Path to the follower URDF (or a MuJoCo XML).
        target_frame_name: Body/frame whose pose FK/IK act on (the tool flange).
        num_dofs: Number of arm joints (default 6). The gripper finger joints in
            the model are left at zero and ignored.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_URDF,
        target_frame_name: str = "tool0",
        num_dofs: int = 6,
        max_step_rad: float | None = None,
        damping: float = 1e-1,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.num_dofs = num_dofs
        self.target_frame_name = target_frame_name
        # Per-call IK trust region (max |Δq| per joint, relative to the seed) and
        # DLS damping. Used as defaults by inverse_kinematics() so the stock
        # LeRobot step — which only forwards orientation_weight — still picks them up.
        self.max_step_rad = max_step_rad
        self.damping = damping

        if model_path.endswith(".urdf"):
            xml_str = self._load_urdf_keep_fixed_frames(model_path)
            self.model = mujoco.MjModel.from_xml_string(xml_str)
        else:
            self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        if self.model.nq < num_dofs:
            raise ValueError(f"Model has {self.model.nq} DoFs but num_dofs={num_dofs}")

        self._ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, target_frame_name
        )
        if self._ee_body_id < 0:
            raise ValueError(
                f"Target frame body {target_frame_name!r} not found in model. "
                f"Bodies: "
                + ", ".join(
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
                    for b in range(self.model.nbody)
                )
            )

        # Joint position limits for IK clamping (arm joints only).
        self.joint_pos_limits = DK1RobotConfig().joint_pos_limits[:num_dofs]

        logger.info(
            "DK1Kinematics loaded: %s (target=%s, %d arm DoF, model nq=%d)",
            model_path,
            target_frame_name,
            num_dofs,
            self.model.nq,
        )

    @staticmethod
    def _load_urdf_keep_fixed_frames(urdf_path: str) -> str:
        """
        Strip meshes (like gravity_comp) and disable MuJoCo's static-body fusion so
        fixed-joint frames (e.g. ``tool0``) survive as addressable bodies.
        """
        xml_str = _urdf_strip_meshes(urdf_path)
        root = ET.fromstring(xml_str)
        mj = ET.SubElement(root, "mujoco")
        ET.SubElement(mj, "compiler", {"fusestatic": "false", "discardvisual": "false"})
        return ET.tostring(root, encoding="unicode")

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------

    def _set_arm_qpos(self, q: np.ndarray) -> None:
        self.data.qpos[: self.num_dofs] = np.asarray(q, dtype=float)[: self.num_dofs]

    def forward_kinematics(self, joint_pos: np.ndarray) -> np.ndarray:
        """
        Compute the target-frame pose for a joint configuration.

        Args:
            joint_pos: Joint positions in radians, shape (num_dofs,) or larger
                (extra entries, e.g. the gripper, are ignored).

        Returns:
            4x4 homogeneous transform of the target frame in the world frame.
        """
        mujoco = self._mujoco
        self._set_arm_qpos(joint_pos)
        mujoco.mj_kinematics(self.model, self.data)

        T = np.eye(4)
        T[:3, :3] = self.data.xmat[self._ee_body_id].reshape(3, 3)
        T[:3, 3] = self.data.xpos[self._ee_body_id]
        return T

    def jacobian(self, joint_pos: np.ndarray) -> np.ndarray:
        """
        Geometric Jacobian (world frame) of the target frame.

        Returns:
            (6, num_dofs) array stacking [linear; angular] velocity Jacobians.
        """
        mujoco = self._mujoco
        self._set_arm_qpos(joint_pos)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        point = self.data.xpos[self._ee_body_id]
        mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self._ee_body_id)
        return np.vstack([jacp[:, : self.num_dofs], jacr[:, : self.num_dofs]])

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 1.0,
        max_iters: int = 100,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-3,
        damping: float | None = None,
        step_scale: float = 1.0,
        max_step: float | None = None,
    ) -> np.ndarray:
        """
        Damped-least-squares inverse kinematics, seeded from the current joints.

        Signature matches LeRobot's ``RobotKinematics.inverse_kinematics`` so this
        class is a drop-in for the stock EE processor steps.

        Args:
            current_joint_pos: Initial guess in radians, shape (num_dofs,) or
                larger (extra entries ignored).
            desired_ee_pose: Target 4x4 homogeneous transform.
            position_weight: Relative weight of the position error.
            orientation_weight: Relative weight of the orientation error. Set low
                (e.g. 0.01) for position-priority IK; 1.0 tracks full 6-DOF pose.
            damping: DLS regularizer λ. Falls back to ``self.damping`` if None.
                Higher = smoother/more stable near singularities, less accurate.
            max_step: Per-call trust region — the returned solution is kept within
                ``±max_step`` (radians, per joint) of ``current_joint_pos``. Falls
                back to ``self.max_step_rad`` if None; None disables it. Applied
                *inside* the loop (not as a post-clamp) so the solver stays on the
                local branch and never chases a far/flipped solution. For this to
                act as a per-step rate limit, seed from the previous command
                (``initial_guess_current_joints=False`` on the LeRobot step).

        Returns:
            Joint positions in radians, shape (num_dofs,), clamped to limits.
        """
        damping = self.damping if damping is None else damping
        max_step = self.max_step_rad if max_step is None else max_step
        q_seed = np.asarray(current_joint_pos, dtype=float)[: self.num_dofs].copy()
        q = q_seed.copy()
        lo = self.joint_pos_limits[:, 0]
        hi = self.joint_pos_limits[:, 1]
        if max_step is not None:
            lo = np.maximum(lo, q_seed - max_step)
            hi = np.minimum(hi, q_seed + max_step)
        p_des = desired_ee_pose[:3, 3]
        R_des = desired_ee_pose[:3, :3]
        w = np.array([position_weight] * 3 + [orientation_weight] * 3)

        for _ in range(max_iters):
            T = self.forward_kinematics(q)
            pos_err = p_des - T[:3, 3]
            rot_err = _rotation_matrix_to_rotvec(R_des @ T[:3, :3].T)

            # Convergence on the true (unweighted) pose error.
            if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err) < rot_tol:
                break

            err = np.concatenate([pos_err, rot_err])
            J = self.jacobian(q)
            # Weighted damped least squares: dq = Jw^T (Jw Jw^T + lambda^2 I)^-1 ew
            Jw = J * w[:, None]
            ew = err * w
            JJt = Jw @ Jw.T + (damping**2) * np.eye(6)
            dq = Jw.T @ np.linalg.solve(JJt, ew)
            # Clamp to joint limits AND the trust region (folded into lo/hi). The
            # cumulative bound keeps total motion per call within ±max_step of the
            # seed, so a far target rate-limits instead of jumping.
            q = np.clip(q + step_scale * dq, lo, hi)

        return q
