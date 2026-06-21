"""EE-space dataset recording for DK1.

Wraps `lerobot-record` with FK on leader joints + follower observation and IK
on the action, so the dataset stores cartesian columns. All `lerobot-record`
CLI flags work here.

    python examples/ee_record.py \\
        --robot.type=dk1_follower --robot.port=/dev/ttyACM1 \\
        --teleop.type=dk1_leader  --teleop.port=/dev/ttyACM0 \\
        --dataset.fps=30 \\
        --dataset.repo_id=felixmin/dk1-pick-cube-ee \\
        --dataset.single_task="pick up the cube and place it in the bin" \\
        --dataset.num_episodes=10 \\
        --dataset.episode_time_s=30 \\
        --dataset.reset_time_s=30 \\
        --dataset.push_to_hub=false
"""

from lerobot.configs import parser
from lerobot.scripts.lerobot_record import RecordConfig, record

from lerobot_robot_trlc_dk1.ee_pipelines import (
    make_dk1_kinematics,
    make_ee_to_follower_joints_pipeline,
    make_follower_obs_to_ee_pipeline,
    make_leader_joints_to_ee_pipeline,
)


@parser.wrap()
def main(cfg: RecordConfig) -> None:
    follower_kin = make_dk1_kinematics()
    leader_kin = make_dk1_kinematics()

    record(
        cfg,
        teleop_action_processor=make_leader_joints_to_ee_pipeline(leader_kin, tuple_input=True),
        robot_observation_processor=make_follower_obs_to_ee_pipeline(follower_kin),
        robot_action_processor=make_ee_to_follower_joints_pipeline(
            follower_kin, converge_ik=False
        ),
    )


if __name__ == "__main__":
    main()
