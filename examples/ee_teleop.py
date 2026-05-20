"""Leader→follower teleop in EE space, via LeRobot's native processor pipeline.

Mirrors `lerobot/examples/so100_to_so100_EE/teleoperate.py`. Each tick:

    leader joints  ──FK──▶  ee.x..wz, ee.gripper_pos  ──IK──▶  follower joints

The leader and follower share the follower URDF (Option A — both are kinematic
twins of the same chain). FK on the leader runs through the follower's link
lengths; IK puts those targets back on the follower. Functionally close to plain
joint-space teleop but routes through the standard LeRobot EE pipeline, so the
same pipeline classes drop into recording / training / rollout unchanged.

Compared to the existing joint-space `examples/teleop.py`, this adds:
  - EE-space safety bounds (workspace clip + per-step jump cap).
  - placo IK every tick instead of forwarding raw joints.
  - Reusable plumbing for recording EE-space datasets.
"""

import argparse
import logging
import time

from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from lerobot_robot_trlc_dk1.ee_pipelines import (
    make_dk1_kinematics,
    make_ee_to_follower_joints_pipeline,
    make_leader_joints_to_ee_pipeline,
)
from lerobot_robot_trlc_dk1.follower import DK1Follower, DK1FollowerConfig
from lerobot_robot_trlc_dk1.leader import DK1Leader, DK1LeaderConfig


logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EE-space teleop for DK1 leader → follower.")
    p.add_argument("--follower-port", default="/dev/ttyACM1")
    p.add_argument("--leader-port", default="/dev/ttyACM0")
    p.add_argument("--fps", type=int, default=30)
    return p.parse_args()


def main() -> None:
    init_logging()
    args = _parse_args()

    follower = DK1Follower(DK1FollowerConfig(port=args.follower_port))
    leader = DK1Leader(DK1LeaderConfig(port=args.leader_port))

    follower_kin = make_dk1_kinematics()
    leader_kin = make_dk1_kinematics()

    leader_to_ee = make_leader_joints_to_ee_pipeline(leader_kin, tuple_input=False)
    ee_to_follower = make_ee_to_follower_joints_pipeline(follower_kin)

    leader.connect()
    follower.connect()
    init_rerun(session_name="dk1_ee_teleop")

    logger.info("Starting EE teleop loop. Ctrl-C to stop.")
    try:
        while True:
            t0 = time.perf_counter()

            robot_obs = follower.get_observation()
            leader_joints = leader.get_action()

            ee_action = leader_to_ee(leader_joints)
            follower_action = ee_to_follower((ee_action, robot_obs))

            follower.send_action(follower_action)

            log_rerun_data(observation=ee_action, action=follower_action)
            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        logger.info("Stopping teleop...")
    finally:
        leader.disconnect()
        follower.disconnect()


if __name__ == "__main__":
    main()
