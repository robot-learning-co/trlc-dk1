"""EE-space pipeline builders for DK1 record / rollout / teleop."""

from pathlib import Path
from typing import Any

from lerobot.processor import (
    RobotProcessorPipeline,
    observation_to_transition,
    robot_action_observation_to_transition,
    robot_action_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)

from lerobot_robot_trlc_dk1.kinematics import DK1RobotKinematics


_REPO_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = str(_REPO_ROOT / "urdf" / "follower" / "TRLC-DK1-Follower.urdf")
TARGET_FRAME = "tool0"
JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
MOTOR_NAMES = JOINT_NAMES + ["gripper"]

# Generous starting bounds — tighten after a calibration session.
EE_BOUNDS = {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}
MAX_EE_STEP_M = 0.10


def make_dk1_kinematics(
    *,
    kinetic_reg: float | None = 1e-4,
    solver_dt: float = 1 / 30,
    ik_max_iters: int = 20,
    ik_pos_tol: float = 1e-3,
    ik_ori_tol: float = 1e-2,
) -> DK1RobotKinematics:
    return DK1RobotKinematics(
        URDF_PATH,
        TARGET_FRAME,
        JOINT_NAMES,
        kinetic_reg=kinetic_reg,
        solver_dt=solver_dt,
        ik_max_iters=ik_max_iters,
        ik_pos_tol=ik_pos_tol,
        ik_ori_tol=ik_ori_tol,
    )


def _action_only_from_tuple(action_observation: tuple[Any, Any]) -> Any:
    action, _observation = action_observation
    return robot_action_to_transition(action)


def make_leader_joints_to_ee_pipeline(
    kinematics: DK1RobotKinematics, *, tuple_input: bool
) -> RobotProcessorPipeline:
    return RobotProcessorPipeline(
        steps=[ForwardKinematicsJointsToEE(kinematics=kinematics, motor_names=MOTOR_NAMES)],
        to_transition=_action_only_from_tuple if tuple_input else robot_action_to_transition,
        to_output=transition_to_robot_action,
    )


def make_follower_obs_to_ee_pipeline(kinematics: DK1RobotKinematics) -> RobotProcessorPipeline:
    return RobotProcessorPipeline(
        steps=[ForwardKinematicsJointsToEE(kinematics=kinematics, motor_names=MOTOR_NAMES)],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )


def make_ee_to_follower_joints_pipeline(
    kinematics: DK1RobotKinematics,
    *,
    safety: bool = True,
    initial_guess_current_joints: bool = True,
    converge_ik: bool = False,
) -> RobotProcessorPipeline:
    kinematics.converge_ik = converge_ik
    steps: list[Any] = []
    if safety:
        steps.append(EEBoundsAndSafety(end_effector_bounds=EE_BOUNDS, max_ee_step_m=MAX_EE_STEP_M))
    steps.append(
        InverseKinematicsEEToJoints(
            kinematics=kinematics,
            motor_names=MOTOR_NAMES,
            initial_guess_current_joints=initial_guess_current_joints,
        )
    )
    return RobotProcessorPipeline(
        steps=steps,
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
