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

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
import serial
import time
import logging
from typing import Any

import numpy as np
import pinocchio as pin
import yaml

from lerobot.cameras import CameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot, RobotConfig
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot_robot_trlc_dk1.motors.DM_Control_Python.DM_CAN import *

logger = logging.getLogger(__name__)


def map_range(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


@dataclass
class JointImpedanceParams:
    """Impedance parameters for a single joint."""
    kp: float  # Stiffness (Nm/rad)
    damping_ratio: float = 1.0  # Critical damping = 1.0

    @property
    def kd(self) -> float:
        """Compute damping: kd = damping_ratio * sqrt(kp)."""
        return self.damping_ratio * np.sqrt(self.kp)


@RobotConfig.register_subclass("dk1_follower")
@dataclass
class DK1FollowerConfig(RobotConfig):
    port: str
    controller_type: str = "pos_vel"  # "pos_vel" | "joint_impedance"
    disable_torque_on_disconnect: bool = False

    # POS_VEL controller parameters
    joint_velocity_scaling: float = 0.2

    # Joint impedance controller parameters (only used when controller_type="joint_impedance")
    urdf_path: str | None = None
    controller_config_path: str | None = None  # Path to per-joint YAML config
    default_kp_large: float = 50.0             # Default kp for DM4340 (joints 1-3)
    default_kp_small: float = 20.0             # Default kp for DM4310 (joints 4-6)
    damping_ratio: float = 1.0                # Global damping ratio (kd = ratio * sqrt(kp))

    # Shared parameters
    max_gripper_torque: float = 1.0  # Nm (/0.00875m spur gear radius = 114N gripper force)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


class DK1Follower(Robot):
    """
    TRLC-DK1 Follower Arm designed by The Robot Learning Company.

    Supports two control modes via the ``controller_type`` config field:

    - ``"pos_vel"`` (default): position + velocity control handled by motor firmware.
    - ``"joint_impedance"``: MIT mode with explicit per-cycle PD gains and gravity
      compensation feedforward computed from a Pinocchio URDF model.

      Control law (executed at motor driver level):
          tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_gravity
    """

    config_class = DK1FollowerConfig
    name = "dk1_follower"

    # Motor torque limits (Nm) — used by impedance mode
    TORQUE_LIMITS = {
        "joint_1": 28.0,  # DM4340
        "joint_2": 28.0,
        "joint_3": 28.0,
        "joint_4": 10.0,  # DM4310
        "joint_5": 10.0,
        "joint_6": 10.0,
    }

    # MIT mode parameter limits (from DM_CAN.py) — used by impedance mode
    MIT_KP_MAX = 500.0
    MIT_KD_MAX = 5.0

    def __init__(self, config: DK1FollowerConfig):
        super().__init__(config)

        # Constants for EMIT control
        self.DM4310_TORQUE_CONSTANT = 0.945  # Nm/A
        self.EMIT_VELOCITY_SCALE = 100  # rad/s
        self.EMIT_CURRENT_SCALE = 1000  # A

        self.JOINT_LIMITS = {
            "joint_4": (-100/180*np.pi, 100/180*np.pi),
            "joint_5": (-90/180*np.pi, 90/180*np.pi),
        }

        self.DM4310_SPEED = 200/60*2*np.pi   # rad/s (200  rpm | 20.94 rad/s)
        self.DM4340_SPEED = 52.5/60*2*np.pi  # rad/s (52.5 rpm | 5.49  rad/s)

        self.config = config
        self.motors = {
            "joint_1": Motor(DM_Motor_Type.DM4340, 0x01, 0x11),
            "joint_2": Motor(DM_Motor_Type.DM4340, 0x02, 0x12),
            "joint_3": Motor(DM_Motor_Type.DM4340, 0x03, 0x13),
            "joint_4": Motor(DM_Motor_Type.DM4310, 0x04, 0x14),
            "joint_5": Motor(DM_Motor_Type.DM4310, 0x05, 0x15),
            "joint_6": Motor(DM_Motor_Type.DM4310, 0x06, 0x16),
            "gripper": Motor(DM_Motor_Type.DM4310, 0x07, 0x17),
        }
        self.joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

        self.control = None
        self.serial_device = None
        self.bus_connected = False

        self.gripper_open_pos = 0.0
        self.gripper_closed_pos = -4.7

        # Pinocchio model — loaded on connect() when controller_type="joint_impedance"
        self.pin_model = None
        self.pin_data = None

        # Per-joint impedance parameters — populated when controller_type="joint_impedance"
        self.params: dict[str, JointImpedanceParams] = {}
        if config.controller_type == "joint_impedance":
            self.params = self._load_impedance_config()

        self.cameras = make_cameras_from_configs(config.cameras)

    # ------------------------------------------------------------------
    # Impedance helpers
    # ------------------------------------------------------------------

    def _load_impedance_config(self) -> dict[str, JointImpedanceParams]:
        """Load per-joint impedance parameters from YAML or fall back to defaults."""
        params: dict[str, JointImpedanceParams] = {}

        if self.config.controller_config_path:
            yaml_path = Path(self.config.controller_config_path)
            if yaml_path.exists():
                with open(yaml_path, "r") as f:
                    yaml_config = yaml.safe_load(f)

                for joint_name in self.joint_names:
                    if joint_name in yaml_config.get("joints", {}):
                        joint_cfg = yaml_config["joints"][joint_name]
                        params[joint_name] = JointImpedanceParams(
                            kp=joint_cfg.get("kp", self.config.default_kp_large),
                            damping_ratio=joint_cfg.get("damping_ratio", self.config.damping_ratio),
                        )

                logger.info(f"Loaded impedance config from {yaml_path}")

        # Fill in defaults for any joints not covered by the YAML
        for joint_name in self.joint_names:
            if joint_name not in params:
                kp = (
                    self.config.default_kp_large
                    if joint_name in ("joint_1", "joint_2", "joint_3")
                    else self.config.default_kp_small
                )
                params[joint_name] = JointImpedanceParams(
                    kp=kp,
                    damping_ratio=self.config.damping_ratio,
                )

        for joint_name, p in params.items():
            logger.info(f"{joint_name}: kp={p.kp:.1f}, kd={p.kd:.2f}")

        return params

    def _load_pinocchio_model(self) -> None:
        """Load URDF into a Pinocchio model."""
        urdf_path = Path(self.config.urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        self.pin_model = pin.buildModelFromUrdf(str(urdf_path))
        self.pin_data = self.pin_model.createData()

        logger.info(f"Loaded Pinocchio model from {urdf_path}")
        logger.info(f"Model has {self.pin_model.nq} DOF")

        # Print joint ordering for verification
        logger.info("Pinocchio joint order:")
        for i, name in enumerate(self.pin_model.names[1:]):  # Skip universe joint
            logger.info(f"  q[{i}] = {name}")

    def compute_feedforward_torque(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """
        Compute feedforward torques (gravity + Coriolis + centripetal) using Pinocchio.

        Args:
            q: Joint positions array matching the Pinocchio model DOF.
            dq: Joint velocities array matching the Pinocchio model DOF.

        Returns:
            Feedforward torques (Nm).
        """
        return pin.nonLinearEffects(self.pin_model, self.pin_data, q, dq)

    # ------------------------------------------------------------------
    # LeRobot Robot interface
    # ------------------------------------------------------------------

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        if self.config.controller_type == "joint_impedance":
            self._load_pinocchio_model()

        self.serial_device = serial.Serial(self.config.port, 921600, timeout=0.5)
        time.sleep(0.5)

        self.control = MotorControl(self.serial_device)
        self.bus_connected = True
        self.configure()

        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        for key, motor in self.motors.items():
            self.control.addMotor(motor)

            for _ in range(3):
                self.control.refresh_motor_status(motor)
                time.sleep(0.01)

            if self.control.read_motor_param(motor, DM_variable.CTRL_MODE) is not None:
                print(f"{key} ({motor.MotorType.name}) is connected.")
                print(f"Controller type: {self.config.controller_type}")

                if key == "gripper":
                    self.control.switchControlMode(motor, Control_Type.Torque_Pos)
                elif self.config.controller_type == "joint_impedance":
                    self.control.switchControlMode(motor, Control_Type.MIT)
                else:
                    self.control.switchControlMode(motor, Control_Type.POS_VEL)

                self.control.enable(motor)
            else:
                raise Exception(f"Unable to read from {key} ({motor.MotorType.name}).")

        # POS_VEL-only: write firmware tuning parameters
        if self.config.controller_type == "pos_vel":
            for joint in ["joint_1", "joint_2", "joint_3"]:
                self.control.change_motor_param(self.motors[joint], DM_variable.ACC, 10.0)
                self.control.change_motor_param(self.motors[joint], DM_variable.DEC, -10.0)
                self.control.change_motor_param(self.motors[joint], DM_variable.KP_APR, 200)
                self.control.change_motor_param(self.motors[joint], DM_variable.KI_APR, 10)

            self.control.change_motor_param(self.motors["gripper"], DM_variable.KP_APR, 100)

        # Open gripper and set zero position (shared by both modes)
        self.control.switchControlMode(self.motors["gripper"], Control_Type.VEL)
        self.control.control_Vel(self.motors["gripper"], 10.0)
        while True:
            self.control.refresh_motor_status(self.motors["gripper"])
            tau = self.motors["gripper"].getTorque()
            if tau > 1.2:
                self.control.control_Vel(self.motors["gripper"], 0.0)
                self.control.disable(self.motors["gripper"])
                self.control.set_zero_position(self.motors["gripper"])
                time.sleep(0.2)
                self.control.enable(self.motors["gripper"])
                break
            time.sleep(0.01)
        self.control.switchControlMode(self.motors["gripper"], Control_Type.Torque_Pos)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()

        obs_dict = {}
        for key, motor in self.motors.items():
            self.control.refresh_motor_status(motor)
            if key == "gripper":
                # Normalize gripper position between 1 (closed) and 0 (open)
                obs_dict[f"{key}.pos"] = map_range(
                    motor.getPosition(), self.gripper_open_pos, self.gripper_closed_pos, 0.0, 1.0)
            else:
                obs_dict[f"{key}.pos"] = motor.getPosition()

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def _send_gripper_action(self, goal_pos: dict[str, float]) -> None:
        """Send gripper command (force-position mode, shared by all controllers)."""
        if "gripper" not in goal_pos:
            return

        self.control.refresh_motor_status(self.motors["gripper"])
        gripper_goal_pos_mapped = map_range(
            goal_pos["gripper"], 0.0, 1.0, self.gripper_open_pos, self.gripper_closed_pos)
        self.control.control_pos_force(
            self.motors["gripper"],
            gripper_goal_pos_mapped,
            self.DM4310_SPEED * self.EMIT_VELOCITY_SCALE,
            i_des=self.config.max_gripper_torque / self.DM4310_TORQUE_CONSTANT * self.EMIT_CURRENT_SCALE,
        )

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        if self.config.controller_type == "joint_impedance":
            return self._send_action_impedance(goal_pos)
        else:
            return self._send_action_pos_vel(goal_pos)

    def _send_action_pos_vel(self, goal_pos: dict[str, float]) -> dict[str, Any]:
        """Send position+velocity commands (POS_VEL firmware mode)."""
        for key, motor in self.motors.items():
            if key == "gripper":
                self._send_gripper_action(goal_pos)
            else:
                if key in self.JOINT_LIMITS:
                    goal_pos[key] = np.clip(goal_pos[key], self.JOINT_LIMITS[key][0], self.JOINT_LIMITS[key][1])

                self.control.control_Pos_Vel(
                    motor, goal_pos[key], self.config.joint_velocity_scaling * self.DM4340_SPEED)

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def _send_action_impedance(self, goal_pos: dict[str, float]) -> dict[str, Any]:
        """
        Send MIT impedance commands with dynamics compensation feedforward.

        Control law (at motor driver level):
            tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff
        where tau_ff includes gravity, Coriolis, and centripetal terms.
        """
        # Read current arm state for dynamics computation
        q_current = np.zeros(self.pin_model.nq)
        dq_current = np.zeros(self.pin_model.nq)
        for i, joint_name in enumerate(self.joint_names):
            self.control.refresh_motor_status(self.motors[joint_name])
            if i < self.pin_model.nq:
                q_current[i] = self.motors[joint_name].getPosition()
                dq_current[i] = self.motors[joint_name].getVelocity()

        tau_ff = self.compute_feedforward_torque(q_current, dq_current)

        # Send commands to all motors
        for key, motor in self.motors.items():
            if key == "gripper":
                self._send_gripper_action(goal_pos)
            elif key in goal_pos:
                # MIT impedance control for arm joints
                i = self.joint_names.index(key)
                imp = self.params[key]

                q_des = goal_pos[key]

                if key in self.JOINT_LIMITS:
                    q_des = np.clip(q_des, self.JOINT_LIMITS[key][0], self.JOINT_LIMITS[key][1])

                kp = min(imp.kp, self.MIT_KP_MAX)
                kd = min(imp.kd, self.MIT_KD_MAX)

                tau_joint_ff = (
                    np.clip(tau_ff[i], -self.TORQUE_LIMITS[key], self.TORQUE_LIMITS[key])
                    if i < len(tau_ff)
                    else 0.0
                )

                self.control.controlMIT(motor, kp=kp, kd=kd, q=q_des, dq=0.0, tau=tau_joint_ff)

                goal_pos[key] = q_des

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.disable_torque_on_disconnect:
            for motor in self.motors.values():
                self.control.disable(motor)
        else:
            self.control.serial_.close()
        self.bus_connected = False

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
