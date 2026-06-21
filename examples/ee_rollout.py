"""EE-space policy rollout for DK1.

Replays `lerobot-rollout`'s body with EE pipelines wired into
`build_rollout_context`. All `lerobot-rollout` CLI flags work here.

    python examples/ee_rollout.py \\
        --robot.type=dk1_follower --robot.port=/dev/ttyACM1 \\
        --robot.cameras="{ context: {type: opencv, index_or_path: 0, width: 640, height: 360, fps: 30, rotation: 180} }" \\
        --policy.path=outputs/train/dk1-pick-cube-diffusion-ee/checkpoints/last/pretrained_model \\
        --device=cuda \\
        --fps=30 \\
        --duration=300 \\
        --task="pick up the cube and place it in the bin"
"""

import logging

# Side-effect imports: register camera configs with draccus, as lerobot_rollout does.
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.rollout import RolloutConfig, build_rollout_context, create_strategy
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun

from lerobot_robot_trlc_dk1.ee_pipelines import (
    make_dk1_kinematics,
    make_ee_to_follower_joints_pipeline,
    make_follower_obs_to_ee_pipeline,
    make_leader_joints_to_ee_pipeline,
)
# Side-effect import: registers --inference.type=sync_relative for relative-action policies.
from lerobot_robot_trlc_dk1 import sync_relative_inference  # noqa: F401


logger = logging.getLogger(__name__)


@parser.wrap()
def main(cfg: RolloutConfig) -> None:
    init_logging()

    if cfg.display_data:
        init_rerun(session_name="dk1_ee_rollout", ip=cfg.display_ip, port=cfg.display_port)

    follower_kin = make_dk1_kinematics()
    leader_kin = make_dk1_kinematics()

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    ctx = build_rollout_context(
        cfg,
        shutdown_event,
        teleop_action_processor=make_leader_joints_to_ee_pipeline(leader_kin, tuple_input=True),
        robot_action_processor=make_ee_to_follower_joints_pipeline(
            follower_kin, safety=True, initial_guess_current_joints=True
        ),
        robot_observation_processor=make_follower_obs_to_ee_pipeline(follower_kin),
    )

    strategy = create_strategy(cfg.strategy)
    logger.info("Rollout strategy: %s | FPS: %.0f | Duration: %s",
                cfg.strategy.type, cfg.fps,
                f"{cfg.duration}s" if cfg.duration > 0 else "infinite")

    try:
        strategy.setup(ctx)
        strategy.run(ctx)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        strategy.teardown(ctx)


if __name__ == "__main__":
    main()
