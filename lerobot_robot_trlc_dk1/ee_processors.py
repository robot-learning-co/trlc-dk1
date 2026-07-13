#   Copyright 2025 The Robot Learning Company UG (haftungsbeschränkt). All rights reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
End-effector (Cartesian) action-representation processors for the DK1 follower.

This is the LeRobot-glue layer: it assembles LeRobot's stock end-effector
processor *steps* (from ``lerobot.robots.so_follower.robot_kinematic_processor``)
into the three ``RobotProcessorPipeline`` objects that ``lerobot`` record/teleop
loops consume, fed with a DK1-native (radians) ``DK1Kinematics`` model.

``DK1Follower.send_action`` stays joint-space; the inverse kinematics runs inside
``robot_action_processor`` just before the robot, so nothing about the robot class
changes.

Switch action representation with a single call::

    teleop_proc, robot_proc, obs_proc = make_dk1_processors(action_space="ee")

The default (``keep_joints=True``) records a *superset* — both ``joint_*.pos`` and
``ee.*`` — so a dataset collected with a joint leader can train either a
joint-space or an EE-space policy. With a joint leader this is lossless (``ee.*``
is forward kinematics of the recorded joints) and needs no IK at record time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.processor import IdentityProcessorStep, RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.processor.pipeline import PipelineFeatureType
from lerobot.utils.rotation import Rotation
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEEAction,
    ForwardKinematicsJointsToEEObservation,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)

from trlc_dk1_control.kinematics import DK1Kinematics

from .follower import JOINT_NAMES

# Sensible defaults; tune per workspace.
DEFAULT_EE_BOUNDS = {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}
DEFAULT_EE_STEP_SIZES = {"x": 1.0, "y": 1.0, "z": 1.0}
DEFAULT_MAX_EE_STEP_M = 0.20

_EE_KEYS = ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]


def _fk_keep_joints(data: dict[str, Any], kinematics: DK1Kinematics, motor_names: list[str]) -> dict[str, Any]:
    """Append ``ee.*`` computed by forward kinematics WITHOUT dropping the joint keys."""
    q = np.array([data[f"{n}.pos"] for n in motor_names], dtype=float)
    t = kinematics.forward_kinematics(q)
    pos = t[:3, 3]
    tw = Rotation.from_matrix(t[:3, :3]).as_rotvec()
    data["ee.x"], data["ee.y"], data["ee.z"] = (float(v) for v in pos)
    data["ee.wx"], data["ee.wy"], data["ee.wz"] = (float(v) for v in tw)
    data["ee.gripper_pos"] = float(data["gripper.pos"])
    return data


def _add_ee_features(features: dict, kind: PipelineFeatureType, feat_type: FeatureType) -> None:
    for k in _EE_KEYS:
        features[kind][f"ee.{k}"] = PolicyFeature(type=feat_type, shape=(1,))


@dataclass
class KeepJointsForwardKinematicsAction(ForwardKinematicsJointsToEEAction):
    """FK on the action that ADDS ``ee.*`` while keeping ``joint_*.pos`` (superset)."""

    def action(self, action):
        return _fk_keep_joints(action, self.kinematics, self.motor_names)

    def transform_features(self, features):
        _add_ee_features(features, PipelineFeatureType.ACTION, FeatureType.ACTION)
        return features


@dataclass
class KeepJointsForwardKinematicsObservation(ForwardKinematicsJointsToEEObservation):
    """FK on the observation that ADDS ``ee.*`` while keeping ``joint_*.pos`` (superset)."""

    def observation(self, observation):
        return _fk_keep_joints(observation, self.kinematics, self.motor_names)

    def transform_features(self, features):
        _add_ee_features(features, PipelineFeatureType.OBSERVATION, FeatureType.STATE)
        return features


def _teleop_pipeline(steps):
    return RobotProcessorPipeline[tuple[dict, dict], dict](
        steps=steps,
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


def _obs_pipeline(steps):
    return RobotProcessorPipeline[dict, dict](
        steps=steps,
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )


def make_dk1_processors(
    action_space: str = "ee",
    teleop_source: str = "joint_leader",
    keep_joints: bool = True,
    kinematics: DK1Kinematics | None = None,
    urdf_path: str | None = None,
    target_frame_name: str = "tool0",
    motor_names: list[str] | None = None,
    end_effector_bounds: dict | None = None,
    end_effector_step_sizes: dict | None = None,
    max_ee_step_m: float = DEFAULT_MAX_EE_STEP_M,
    drive_via_ik: bool = False,
    ik_orientation_weight: float = 1.0,
    ik_max_step_rad: float = 0.1,
    ik_seed_previous: bool = True,
) -> tuple[RobotProcessorPipeline, RobotProcessorPipeline, RobotProcessorPipeline]:
    """
    Build (teleop_action_processor, robot_action_processor, robot_observation_processor).

    Args:
        action_space: ``"joint"`` → identity pipelines (current behavior).
            ``"ee"`` → Cartesian pipelines.
        teleop_source: how the recorded action is produced (only relevant for ``ee``):
            ``"joint_leader"`` — leader emits joint targets; add ``ee.*`` via FK.
            ``"ee_delta"`` — a delta device (spacemouse/phone) emits deltas; build
              EE pose then IK to joints.
            ``"policy"`` — no teleop; policy emits ``ee.*`` and IK converts to joints.
        keep_joints: record the superset (both ``joint_*.pos`` and ``ee.*``). Only
            affects the ``joint_leader`` teleop path and the observation pipeline.
        drive_via_ik: for ``joint_leader``, close the loop through Cartesian space —
            command the follower with IK of the leader's EE pose (``FK → ee.* →
            IK``) instead of forwarding the leader's joints directly. This is the
            genuine EE-control path (and what policy inference uses); it also
            surfaces IK/singularity behavior. When False (default) with
            ``keep_joints=True`` the follower gets the leader's joints verbatim —
            exact and singularity-free, but does not exercise IK.
        ik_orientation_weight: DLS orientation-vs-position weight for the IK step
            (1.0 = full 6-DOF pose; DK1 is 6-DOF so this is well-posed).
        ik_max_step_rad: per-call IK trust region (max |Δq| per joint). Bounds how
            far the IK solution can move each control step — an *in-solver* rate
            limit that keeps the solver on the local branch and prevents jumps.
            Pair with ``ik_seed_previous=True`` so the bound is relative to the
            previous command. Roughly ``max_joint_speed / loop_hz``. None disables.
        ik_seed_previous: seed IK from the previous IK solution instead of the
            follower's (compliant, lagging) measured joints. Improves continuity
            and is required for ``ik_max_step_rad`` to act as a true rate limit.
        kinematics: a ``DK1Kinematics``; created from ``urdf_path``/``target_frame_name``
            if omitted.
    """
    if action_space not in ("joint", "ee"):
        raise ValueError(f"action_space must be 'joint' or 'ee', got {action_space!r}")

    if action_space == "joint":
        return (
            _teleop_pipeline([IdentityProcessorStep()]),
            _teleop_pipeline([IdentityProcessorStep()]),
            _obs_pipeline([IdentityProcessorStep()]),
        )

    if kinematics is None:
        kwargs = {"target_frame_name": target_frame_name}
        if urdf_path is not None:
            kwargs["model_path"] = urdf_path
        kinematics = DK1Kinematics(**kwargs)
    if ik_max_step_rad is not None:
        # In-solver per-step rate limit (applies whether kinematics was built here
        # or passed in); the stock IK step forwards no such arg, so set it here.
        kinematics.max_step_rad = ik_max_step_rad

    arm_motor_names = motor_names or list(JOINT_NAMES)
    # The IK step writes gripper.pos only if "gripper" is in its motor_names.
    ik_motor_names = arm_motor_names + ["gripper"]
    bounds = end_effector_bounds or DEFAULT_EE_BOUNDS
    step_sizes = end_effector_step_sizes or DEFAULT_EE_STEP_SIZES

    # Observation: joints -> add ee.* (keep joints unless asked otherwise).
    obs_fk = (
        KeepJointsForwardKinematicsObservation(kinematics=kinematics, motor_names=arm_motor_names)
        if keep_joints
        else ForwardKinematicsJointsToEEObservation(kinematics=kinematics, motor_names=arm_motor_names)
    )
    robot_observation_processor = _obs_pipeline([obs_fk])

    # DK1 is 6-DOF, so full-pose IK is well-posed; keep orientation weight high
    # (the stock default 0.01 is tuned for 5-DOF SO arms).
    ik_step = InverseKinematicsEEToJoints(
        kinematics=kinematics,
        motor_names=ik_motor_names,
        initial_guess_current_joints=not ik_seed_previous,
        orientation_weight=ik_orientation_weight,
    )

    if teleop_source == "joint_leader":
        # Leader already emits joint targets: record joints (+ee.* via FK), send joints as-is.
        teleop_fk = (
            KeepJointsForwardKinematicsAction(kinematics=kinematics, motor_names=arm_motor_names)
            if keep_joints
            else ForwardKinematicsJointsToEEAction(kinematics=kinematics, motor_names=arm_motor_names)
        )
        teleop_action_processor = _teleop_pipeline([teleop_fk])
        # Drive via IK (true EE closed loop) when requested, or whenever joints were
        # not kept (then they must be reconstructed from ee.*). Otherwise the leader's
        # joints are already present and forwarded verbatim.
        robot_action_processor = _teleop_pipeline(
            [IdentityProcessorStep()] if (keep_joints and not drive_via_ik) else [ik_step]
        )
    elif teleop_source == "ee_delta":
        teleop_action_processor = _teleop_pipeline(
            [
                EEReferenceAndDelta(
                    kinematics=kinematics,
                    end_effector_step_sizes=step_sizes,
                    motor_names=arm_motor_names,
                    use_latched_reference=False,
                ),
                EEBoundsAndSafety(end_effector_bounds=bounds, max_ee_step_m=max_ee_step_m),
                GripperVelocityToJoint(clip_min=0.0, clip_max=1.0),
            ]
        )
        robot_action_processor = _teleop_pipeline([ik_step])
    elif teleop_source == "policy":
        teleop_action_processor = _teleop_pipeline([IdentityProcessorStep()])
        robot_action_processor = _teleop_pipeline([ik_step])
    else:
        raise ValueError(
            f"teleop_source must be 'joint_leader', 'ee_delta' or 'policy', got {teleop_source!r}"
        )

    return teleop_action_processor, robot_action_processor, robot_observation_processor
