"""Leader→follower teleop in EE space (FK on leader, IK on follower).

    python examples/ee_teleop.py \\
        --robot.type=dk1_follower --robot.port=/dev/ttyACM1 \\
        --teleop.type=dk1_leader  --teleop.port=/dev/ttyACM0 \\
        --fps=30
"""

import time

from lerobot.configs import parser
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from lerobot_robot_trlc_dk1.ee_pipelines import (
    make_dk1_kinematics,
    make_ee_to_follower_joints_pipeline,
    make_leader_joints_to_ee_pipeline,
)


@parser.wrap()
def main(cfg: TeleoperateConfig) -> None:
    follower = make_robot_from_config(cfg.robot)
    leader = make_teleoperator_from_config(cfg.teleop)

    follower_kin = make_dk1_kinematics()
    leader_kin = make_dk1_kinematics()

    leader_to_ee = make_leader_joints_to_ee_pipeline(leader_kin, tuple_input=False)
    ee_to_follower = make_ee_to_follower_joints_pipeline(follower_kin)

    leader.connect()
    follower.connect()
    init_rerun(session_name="dk1_ee_teleop")

    try:
        while True:
            t0 = time.perf_counter()

            robot_obs = follower.get_observation()
            leader_joints = leader.get_action()

            ee_action = leader_to_ee(leader_joints)
            follower_action = ee_to_follower((ee_action, robot_obs))

            follower.send_action(follower_action)

            log_rerun_data(observation=ee_action, action=follower_action)
            precise_sleep(max(1.0 / cfg.fps - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("\nStopping teleop...")
    finally:
        leader.disconnect()
        follower.disconnect()


if __name__ == "__main__":
    main()
