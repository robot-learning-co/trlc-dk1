"""
DK1Robot — high-level control interface for the TRLC-DK1 arm.

Architecture:

    User code (any Hz)
        command_joint_pos(q_des)   →  updates shared command buffer
        get_joint_state()          →  reads from shared state buffer

    Server thread (~300 Hz):
        1. Read pos/vel/torque from motor chain
        2. Watchdog: keep holding the last commanded target if no command
           received recently (full stiffness + gravity comp)
        3. Compute gravity compensation torque (MuJoCo)
        4. Enforce safety: clip position, clip torque, over-current counter
        5. Push updated MIT commands to motor chain

    Motor thread (250 Hz) — lives inside DK1MotorChain:
        controlMIT for arm joints + EMIT for gripper
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from .config import DK1RobotConfig, DK1_DEFAULT_CONFIG, DM4310_DQ_MAX
from .gravity_comp import GravityCompensator, NoGravityComp
from .motor_chain import DK1MotorChain

logger = logging.getLogger(__name__)

# Position limit buffer: don't command within this margin of the hard limit (rad)
LIMIT_BUFFER = 0.05


class DK1Robot:
    """
    Standalone control stack for the TRLC-DK1 6-DOF arm + gripper.

    Example::

        cfg = DK1_DEFAULT_CONFIG("/dev/ttyUSB0", mjcf_path="robot.xml")
        robot = DK1Robot(cfg)
        robot.connect()

        q = robot.get_joint_state()["pos"]
        robot.command_joint_pos(q)  # hold current position
        robot.command_gripper(0.0)  # open gripper

        robot.disconnect()
    """

    def __init__(self, config: DK1RobotConfig) -> None:
        self._config = config
        self._motor_chain = DK1MotorChain(config)

        if config.mjcf_path:
            self._grav_comp: GravityCompensator | NoGravityComp = GravityCompensator(
                config.mjcf_path
            )
        else:
            self._grav_comp = NoGravityComp()
            logger.warning("Gravity compensation disabled (no mjcf_path provided)")

        # Server thread shared state — protected by _cmd_lock
        self._cmd_lock = threading.Lock()
        self._q_des = np.zeros(6)      # (6,) arm joint targets in radians
        self._gripper_des = 0.0        # normalised gripper position [0=open, 1=closed]
        self._last_cmd_time: float = 0.0
        self._damping_mode: bool = False   # set when over-current threshold exceeded
        self._stale_warned: bool = False   # throttle the stale-command warning

        # Safety counters
        self._overcurrent_count: int = 0

        self._server_running = False
        self._server_thread: threading.Thread | None = None

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def connect(self) -> None:
        """Start motor chain and server thread. Blocks until motors are ready."""
        self._motor_chain.start()

        # Initialise desired position to current measured position (no startup jump)
        pos, _, _ = self._motor_chain.get_state()
        with self._cmd_lock:
            self._q_des = pos[:6].copy()
            self._last_cmd_time = time.monotonic()

        self._server_running = True
        self._server_thread = threading.Thread(
            target=self._server_loop, daemon=True, name="dk1-server"
        )
        self._server_thread.start()
        logger.info("DK1Robot connected")

    def disconnect(self) -> None:
        """Stop server thread and motor chain."""
        self._server_running = False
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
        self._motor_chain.stop()
        logger.info("DK1Robot disconnected")

    # -------------------------------------------------------------------------
    # Command interface (thread-safe, non-blocking)
    # -------------------------------------------------------------------------

    def command_joint_pos(self, q_des: np.ndarray) -> None:
        """
        Set target joint positions for the 6 arm joints.

        Args:
            q_des: Target positions in radians, shape (6,).
        """
        self._motor_chain.raise_if_failed()
        if q_des.shape != (6,):
            raise ValueError(f"Expected shape (6,), got {q_des.shape}")
        with self._cmd_lock:
            self._q_des = q_des.copy()
            self._last_cmd_time = time.monotonic()
            self._damping_mode = False  # reset on new command
            self._stale_warned = False  # fresh command — re-arm the stale warning

    def command_gripper(self, normalized_pos: float) -> None:
        """
        Set gripper target position.

        Args:
            normalized_pos: 0.0 = fully open, 1.0 = fully closed.
        """
        self._motor_chain.raise_if_failed()
        normalized_pos = float(np.clip(normalized_pos, 0.0, 1.0))
        with self._cmd_lock:
            self._gripper_des = normalized_pos

    # -------------------------------------------------------------------------
    # State interface (thread-safe)
    # -------------------------------------------------------------------------

    def get_joint_state(self) -> dict[str, np.ndarray]:
        """
        Return current arm joint state (excludes gripper).

        Returns:
            dict with keys 'pos', 'vel', 'torque', each shape (6,).
        """
        self._motor_chain.raise_if_failed()
        pos, vel, torque = self._motor_chain.get_state()
        return {
            "pos": pos[:6].copy(),
            "vel": vel[:6].copy(),
            "torque": torque[:6].copy(),
        }

    def get_gripper_state(self) -> dict[str, float]:
        """Return normalised gripper position and torque."""
        self._motor_chain.raise_if_failed()
        pos, _, torque = self._motor_chain.get_state()
        cfg = self._config
        # Linear map open->0, closed->1.  np.interp can't be used here because it
        # requires increasing xp, but gripper_open_pos > gripper_closed_pos.
        normalized = (pos[6] - cfg.gripper_open_pos) / (
            cfg.gripper_closed_pos - cfg.gripper_open_pos
        )
        return {"pos": float(np.clip(normalized, 0.0, 1.0)), "torque": float(torque[6])}

    # -------------------------------------------------------------------------
    # Server loop (~300 Hz)
    # -------------------------------------------------------------------------

    def _server_loop(self) -> None:
        cfg = self._config
        period = 1.0 / cfg.server_thread_hz
        loop_count = 0
        last_log = time.monotonic()

        while self._server_running:
            t_start = time.monotonic()

            pos, vel, torque = self._motor_chain.get_state()

            with self._cmd_lock:
                q_des = self._q_des.copy()
                gripper_des = self._gripper_des
                last_cmd = self._last_cmd_time
                damping = self._damping_mode

            # ------------------------------------------------------------------
            # Watchdog: if no command arrives for command_timeout_s, keep holding
            # the LAST commanded target (full stiffness + gravity comp) instead of
            # resetting to the current position. Resetting to current position made
            # the arm go soft and track its own droop, so it sagged under an
            # unmodelled end-effector payload during long inference gaps (e.g. a
            # diffusion policy's denoising step). Holding the last target freezes
            # the arm in place until the next command. q_des already carries the
            # last commanded target, so we leave it untouched and only warn.
            # ------------------------------------------------------------------
            now = time.monotonic()
            if now - last_cmd > cfg.command_timeout_s and not self._stale_warned:
                logger.warning(
                    "No command for %.2f s — holding last commanded target.",
                    now - last_cmd,
                )
                self._stale_warned = True

            # ------------------------------------------------------------------
            # Gravity compensation
            # ------------------------------------------------------------------
            tau_ff = self._grav_comp.compute(pos[:6]) * cfg.gravity_comp_scale

            # ------------------------------------------------------------------
            # Safety: joint position clamping (with buffer)
            # ------------------------------------------------------------------
            lims = cfg.joint_pos_limits
            q_des_safe = np.clip(
                q_des,
                lims[:, 0] + LIMIT_BUFFER,
                lims[:, 1] - LIMIT_BUFFER,
            )

            # ------------------------------------------------------------------
            # Safety: torque limit clipping
            # ------------------------------------------------------------------
            tau_ff_safe = np.clip(tau_ff, -cfg.joint_torque_limits, cfg.joint_torque_limits)

            # ------------------------------------------------------------------
            # Over-current detection
            # ------------------------------------------------------------------
            over_limit = np.abs(torque[:6]) > cfg.joint_torque_limits
            if np.any(over_limit):
                self._overcurrent_count += 1
                if self._overcurrent_count >= cfg.overcurrent_threshold:
                    logger.warning(
                        "Over-current threshold reached (joints %s). Entering damping mode.",
                        np.where(over_limit)[0] + 1,
                    )
                    damping = True
                    with self._cmd_lock:
                        self._damping_mode = True
            else:
                self._overcurrent_count = max(0, self._overcurrent_count - 1)

            # In damping mode: zero stiffness, zero feedforward — only kd damping
            if damping:
                kp = np.zeros(6)
                tau_ff_safe = np.zeros(6)
                # Hold current position as target to prevent drift when exiting damping
                q_des_safe = pos[:6].copy()
            else:
                kp = cfg.arm_kp

            kd = cfg.arm_kd
            dq_des = np.zeros(6)

            # ------------------------------------------------------------------
            # Push to motor chain
            # ------------------------------------------------------------------
            self._motor_chain.set_arm_commands(kp, kd, q_des_safe, dq_des, tau_ff_safe)

            # Gripper command
            gripper_q = float(np.interp(
                gripper_des,
                [0.0, 1.0],
                [cfg.gripper_open_pos, cfg.gripper_closed_pos],
            ))
            gripper_vel = DM4310_DQ_MAX * cfg.EMIT_VELOCITY_SCALE
            gripper_i_des = (
                cfg.max_gripper_torque_nm
                / cfg.DM4310_TORQUE_CONSTANT
                * cfg.EMIT_CURRENT_SCALE
            )
            self._motor_chain.set_gripper_command(gripper_q, gripper_vel, gripper_i_des)

            # ------------------------------------------------------------------
            # Maintain period + periodic logging
            # ------------------------------------------------------------------
            elapsed = time.monotonic() - t_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            loop_count += 1
            now = time.monotonic()
            if now - last_log >= 5.0:
                hz = loop_count / (now - last_log)
                logger.debug(
                    "[server] %6.1f Hz  (target %.0f Hz)  loop=%.2f ms",
                    hz,
                    cfg.server_thread_hz,
                    elapsed * 1e3,
                )
                loop_count = 0
                last_log = now
