"""EE-space dataset recording for DK1.

Thin wrapper around `lerobot.scripts.lerobot_record.record` that installs the
EE pipelines (FK on leader joints and follower observation, IK on action) so
the resulting dataset stores cartesian columns:

    observation.state.ee.{x,y,z,wx,wy,wz,gripper_pos}
    action.ee.{x,y,z,wx,wy,wz,gripper_pos}

All CLI flags supported by `lerobot-record` work here too — `@parser.wrap()`
re-exposes the same `RecordConfig`. Example invocation:

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

For policy-driven recording (the `--policy.path=...` case) use ee_rollout.py
instead — `lerobot-record` is teleop-only.
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
    # Leader and follower share the follower URDF (kinematic twins, Option A).
    follower_kin = make_dk1_kinematics()
    leader_kin = make_dk1_kinematics()

    record(
        cfg,
        teleop_action_processor=make_leader_joints_to_ee_pipeline(leader_kin, tuple_input=True),
        robot_observation_processor=make_follower_obs_to_ee_pipeline(follower_kin),
        robot_action_processor=make_ee_to_follower_joints_pipeline(follower_kin),
    )


if __name__ == "__main__":
    main()
