"""EE-space policy rollout for DK1.

Replays the body of `lerobot.scripts.lerobot_rollout.rollout` with our EE
pipelines wired into `build_rollout_context`. The stock `rollout()` calls
`build_rollout_context(cfg, shutdown_event)` with no processor kwargs, which
defaults to identity pipelines — that would feed the policy joint-space
observations and try to send its EE-space outputs straight to the robot. Both
shapes wrong.

`teleop_action_processor` is also required even though there's no teleop in
non-DAgger rollout: `build_rollout_context` runs its `transform_features` to
derive the policy/dataset action names (see `lerobot/rollout/context.py:288`).
Without it, action names default to `joint_*.pos` and the IK step downstream
fails looking for `ee.x`/`ee.y`/...

Re-exposes the standard `RolloutConfig` via `@parser.wrap()`, so all CLI flags
supported by `lerobot-rollout` work here. Example:

    python examples/ee_rollout.py \\
        --robot.type=dk1_follower --robot.port=/dev/ttyACM1 \\
        --robot.cameras="{ context: {type: opencv, index_or_path: 0, width: 640, height: 360, fps: 30, rotation: 180} }" \\
        --policy.path=outputs/train/dk1-pick-cube-act-ee/checkpoints/020000/pretrained_model \\
        --policy.device=mps \\
        --policy.temporal_ensemble_coeff=0.01 \\
        --fps=30 \\
        --duration=300 \\
        --task="pick up the cube and place it in the bin"
"""

import logging

# Side-effect imports: register camera configs with draccus. The stock
# `lerobot.scripts.lerobot_rollout` does the same; we bypass that script, so
# without these `--robot.cameras="{... type: opencv ...}"` fails to parse.
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


logger = logging.getLogger(__name__)


@parser.wrap()
def main(cfg: RolloutConfig) -> None:
    init_logging()

    if cfg.display_data:
        logger.info("Initializing Rerun visualization")
        init_rerun(session_name="dk1_ee_rollout", ip=cfg.display_ip, port=cfg.display_port)

    follower_kin = make_dk1_kinematics()
    leader_kin = make_dk1_kinematics()

    # Re-seed IK from current joints each tick: policy can produce larger jumps
    # than leader-driven teleop, so the actual robot state is a safer seed than
    # the previous IK solution.
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    logger.info("Building rollout context with EE pipelines...")
    ctx = build_rollout_context(
        cfg,
        shutdown_event,
        teleop_action_processor=make_leader_joints_to_ee_pipeline(leader_kin, tuple_input=True),
        robot_action_processor=make_ee_to_follower_joints_pipeline(
            follower_kin, safety=False, initial_guess_current_joints=True
        ),
        robot_observation_processor=make_follower_obs_to_ee_pipeline(follower_kin),
    )

    strategy = create_strategy(cfg.strategy)
    logger.info("Rollout strategy: %s", cfg.strategy.type)
    logger.info(
        "Robot: %s | FPS: %.0f | Duration: %s",
        cfg.robot.type if cfg.robot else "?",
        cfg.fps,
        f"{cfg.duration}s" if cfg.duration > 0 else "infinite",
    )

    try:
        strategy.setup(ctx)
        logger.info("Rollout setup complete, starting rollout...")
        strategy.run(ctx)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        strategy.teardown(ctx)

    logger.info("Rollout finished")


if __name__ == "__main__":
    main()
