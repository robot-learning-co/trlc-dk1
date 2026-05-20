"""EE-space pipeline builders for DK1.

Wraps LeRobot's `RobotProcessorPipeline` with the right converters and steps
for DK1's three EE workflows (teleop / record / rollout) so the example scripts
can't drift on URDF paths, joint names, EE safety bounds, or converter shapes.

The three pipelines are:

    leader joints      ──FK──▶  ee.*           (teleop_action_processor)
    follower joints    ──FK──▶  observation.state.ee.*  (robot_observation_processor)
    ee.* + follower obs ─IK─▶  follower joints (robot_action_processor)

`record` and DAgger-rollout strategies invoke `teleop_action_processor((act, obs))`
with a tuple — see `lerobot/scripts/lerobot_record.py:292` and `:302`,
`lerobot/rollout/strategies/dagger.py:488`. The leader-FK pipeline therefore
needs a tuple-aware `to_transition` that drops the observation; otherwise
`ForwardKinematicsJointsToEE` would also FK the follower observation through
the leader kinematics (a wasted placo solve, since `transition_to_robot_action`
discards the observation anyway).

Plain teleop (`examples/ee_teleop.py`) calls the pipeline with a single action
dict, so it uses the standard single-arg converter.
"""

from __future__ import annotations

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


def make_dk1_kinematics() -> DK1RobotKinematics:
    return DK1RobotKinematics(URDF_PATH, TARGET_FRAME, JOINT_NAMES)


def _action_only_from_tuple(action_observation: tuple[Any, Any]) -> Any:
    action, _observation = action_observation
    return robot_action_to_transition(action)


def make_leader_joints_to_ee_pipeline(
    kinematics: DK1RobotKinematics, *, tuple_input: bool
) -> RobotProcessorPipeline:
    """FK on leader joints → EE-space action.

    `tuple_input=True` for record/rollout (LeRobot hands a `(act, obs)` tuple).
    `tuple_input=False` for plain teleop (caller passes the action dict directly).
    """
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
    initial_guess_current_joints: bool = False,
) -> RobotProcessorPipeline:
    """IK on EE-space action → follower joint command.

    `safety=False` for rollout: `EEBoundsAndSafety` raises on out-of-bounds,
    which would abort the run if the policy ever predicts slightly beyond.

    `initial_guess_current_joints=False` (the default) seeds IK with the
    previous solution — smooth, appropriate for continuous leader-driven
    motion. `True` re-seeds from the robot's current joints every tick, which
    is safer if the upstream command can jump (e.g. policy rollout).
    """
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
