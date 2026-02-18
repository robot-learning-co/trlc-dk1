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
Joint Impedance Controller for TRLC-DK1 Follower Arm.

Implements PD control with gravity compensation feedforward using:
- MIT control mode on DaMiao motors (high-frequency PD at motor driver level)
- Pinocchio for gravity torque computation
- YAML configuration for per-joint stiffness parameters
"""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
import numpy as np
import serial
import time
import logging
import yaml
from typing import Any

import pinocchio as pin

from lerobot.cameras import CameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot, RobotConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from trlc_dk1.motors.DM_Control_Python.DM_CAN import (
    Motor, MotorControl, DM_Motor_Type, Control_Type, DM_variable
)

logger = logging.getLogger(__name__)


def map_range(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Map value from one range to another."""
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


@RobotConfig.register_subclass("dk1_impedance_follower")
@dataclass
class DK1ImpedanceFollowerConfig(RobotConfig):
    """Configuration for DK1 Impedance Follower."""
    port: str
    urdf_path: str
    impedance_config_path: str | None = None  # Path to YAML config

    # Default impedance parameters (used if no YAML or joint not in YAML)
    default_kp_large: float = 50.0   # For DM4340 (joints 1-3)
    default_kp_small: float = 20.0   # For DM4310 (joints 4-6)
    damping_ratio: float = 1.0       # Critical damping

    disable_torque_on_disconnect: bool = False
    max_gripper_torque: float = 1.0  # Nm
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


class DK1ImpedanceFollower(Robot):
    """
    TRLC-DK1 Follower Arm with Joint Impedance Control.

    Uses MIT control mode on DaMiao motors for PD control and
    Pinocchio for gravity compensation torque feedforward.

    Control law (executed at motor driver level):
        tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_gravity

    Works with both teleoperation and inference modes.
    """

    config_class = DK1ImpedanceFollowerConfig
    name = "dk1_impedance_follower"

    # Motor torque limits (Nm)
    TORQUE_LIMITS = {
        "joint_1": 28.0,  # DM4340
        "joint_2": 28.0,
        "joint_3": 28.0,
        "joint_4": 10.0,  # DM4310
        "joint_5": 10.0,
        "joint_6": 10.0,
    }

    # MIT mode parameter limits (from DM_CAN.py)
    MIT_KP_MAX = 500.0
    MIT_KD_MAX = 5.0

    def __init__(self, config: DK1ImpedanceFollowerConfig):
        super().__init__(config)
        self.config = config

        # Motor constants
        self.DM4310_TORQUE_CONSTANT = 0.945  # Nm/A
        self.EMIT_VELOCITY_SCALE = 100
        self.EMIT_CURRENT_SCALE = 1000

        self.DM4310_SPEED = 200 / 60 * 2 * np.pi   # rad/s (200 rpm)
        self.DM4340_SPEED = 52.5 / 60 * 2 * np.pi  # rad/s (52.5 rpm)

        self.JOINT_LIMITS = {
            "joint_4": (-100 / 180 * np.pi, 100 / 180 * np.pi),
            "joint_5": (-90 / 180 * np.pi, 90 / 180 * np.pi),
        }

        # Motor definitions
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

        # Pinocchio model (loaded on connect)
        self.pin_model = None
        self.pin_data = None

        # Load impedance configuration
        self.impedance_params = self._load_impedance_config()

        self.cameras = make_cameras_from_configs(config.cameras)

    def _load_impedance_config(self) -> dict[str, JointImpedanceParams]:
        """Load impedance parameters from YAML or use defaults."""
        params = {}

        if self.config.impedance_config_path:
            yaml_path = Path(self.config.impedance_config_path)
            if yaml_path.exists():
                with open(yaml_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)

                for joint_name in self.joint_names:
                    if joint_name in yaml_config.get('joints', {}):
                        joint_cfg = yaml_config['joints'][joint_name]
                        params[joint_name] = JointImpedanceParams(
                            kp=joint_cfg.get('kp', self.config.default_kp_large),
                            damping_ratio=joint_cfg.get('damping_ratio', self.config.damping_ratio)
                        )

                logger.info(f"Loaded impedance config from {yaml_path}")

        # Fill in defaults for any missing joints
        for joint_name in self.joint_names:
            if joint_name not in params:
                if joint_name in ["joint_1", "joint_2", "joint_3"]:
                    kp = self.config.default_kp_large
                else:
                    kp = self.config.default_kp_small
                params[joint_name] = JointImpedanceParams(
                    kp=kp,
                    damping_ratio=self.config.damping_ratio
                )

        # Log final parameters
        for joint_name, p in params.items():
            logger.info(f"{joint_name}: kp={p.kp:.1f}, kd={p.kd:.2f}")

        return params

    def _load_pinocchio_model(self):
        """Load URDF into Pinocchio model."""
        urdf_path = Path(self.config.urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        self.pin_model = pin.buildModelFromUrdf(str(urdf_path))
        self.pin_data = self.pin_model.createData()

        logger.info(f"Loaded Pinocchio model from {urdf_path}")
        logger.info(f"Model has {self.pin_model.nq} DOF")

    def compute_gravity_compensation(self, q: np.ndarray) -> np.ndarray:
        """
        Compute gravity compensation torques using Pinocchio.

        Args:
            q: Joint positions array matching Pinocchio model DOF

        Returns:
            Gravity compensation torques
        """
        pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q)
        return self.pin_data.g.copy()

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
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

        # Load Pinocchio model
        self._load_pinocchio_model()

        # Connect to serial bus
        self.serial_device = serial.Serial(self.config.port, 921600, timeout=0.5)
        time.sleep(0.5)

        self.control = MotorControl(self.serial_device)
        self.bus_connected = True
        self.configure()

        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected with impedance control")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        """Configure motors: MIT mode for arm joints, Torque_Pos for gripper."""

        for key, motor in self.motors.items():
            self.control.addMotor(motor)

            for _ in range(3):
                self.control.refresh_motor_status(motor)
                time.sleep(0.01)

            if self.control.read_motor_param(motor, DM_variable.CTRL_MODE) is not None:
                print(f"{key} ({motor.MotorType.name}) is connected.")

                if key == "gripper":
                    self.control.switchControlMode(motor, Control_Type.Torque_Pos)
                else:
                    # Arm joints use MIT mode for impedance control
                    self.control.switchControlMode(motor, Control_Type.MIT)

                self.control.enable(motor)
            else:
                raise Exception(f"Unable to read from {key} ({motor.MotorType.name}).")

        # Initialize gripper (open fully and set zero position)
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

        start = time.perf_counter()
        obs_dict = {}

        for key, motor in self.motors.items():
            self.control.refresh_motor_status(motor)
            if key == "gripper":
                obs_dict[f"{key}.pos"] = map_range(
                    motor.getPosition(), self.gripper_open_pos, self.gripper_closed_pos, 0.0, 1.0)
            else:
                obs_dict[f"{key}.pos"] = motor.getPosition()

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send impedance-controlled action to the robot.

        Uses MIT mode: tau = kp*(q_des-q) + kd*(dq_des-dq) + tau_ff
        where tau_ff is gravity compensation from Pinocchio.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Get current joint positions for gravity compensation
        q_current = np.zeros(self.pin_model.nq)
        for i, joint_name in enumerate(self.joint_names):
            self.control.refresh_motor_status(self.motors[joint_name])
            if i < self.pin_model.nq:
                q_current[i] = self.motors[joint_name].getPosition()

        # Compute gravity compensation torques
        tau_gravity = self.compute_gravity_compensation(q_current)

        # Send commands to arm joints using MIT mode
        for i, joint_name in enumerate(self.joint_names):
            motor = self.motors[joint_name]
            imp = self.impedance_params[joint_name]

            # Get desired position (default to current if not specified)
            q_des = goal_pos.get(joint_name, motor.getPosition())

            # Apply joint limits
            if joint_name in self.JOINT_LIMITS:
                q_des = np.clip(q_des, self.JOINT_LIMITS[joint_name][0], self.JOINT_LIMITS[joint_name][1])

            # Get impedance parameters (clamp to motor limits)
            kp = min(imp.kp, self.MIT_KP_MAX)
            kd = min(imp.kd, self.MIT_KD_MAX)

            # Gravity feedforward torque (saturate to motor limits)
            if i < len(tau_gravity):
                tau_ff = np.clip(tau_gravity[i], -self.TORQUE_LIMITS[joint_name], self.TORQUE_LIMITS[joint_name])
            else:
                tau_ff = 0.0

            # MIT control: tau = kp*(q_des-q) + kd*(dq_des-dq) + tau_ff
            self.control.controlMIT(
                motor,
                kp=kp,
                kd=kd,
                q=q_des,      # Desired position
                dq=0.0,       # Desired velocity (0 for position hold)
                tau=tau_ff    # Gravity feedforward
            )

            goal_pos[joint_name] = q_des

        # Handle gripper (force-position mode, unchanged from DK1Follower)
        if "gripper" in goal_pos:
            self.control.refresh_motor_status(self.motors["gripper"])
            gripper_goal_pos_mapped = map_range(
                goal_pos["gripper"], 0.0, 1.0, self.gripper_open_pos, self.gripper_closed_pos)
            self.control.control_pos_force(
                self.motors["gripper"],
                gripper_goal_pos_mapped,
                self.DM4310_SPEED * self.EMIT_VELOCITY_SCALE,
                i_des=self.config.max_gripper_torque / self.DM4310_TORQUE_CONSTANT * self.EMIT_CURRENT_SCALE
            )

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
