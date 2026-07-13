"""
Collect a dataset with an end-effector (Cartesian) action representation.

Uses LeRobot's ``record()`` — which owns the robot, teleop, cameras, keyboard
episode control (right arrow = end episode, left = re-record, ESC = stop), dataset
creation and (optional) hub upload — and just injects our three EE pipelines.
Nothing about ``DK1Follower`` changes; the recorded action space is chosen entirely
by the processors.

Because the DK1 leader is joint-matched to the follower, this records with
``drive_via_ik=False``: the follower is driven with the leader's joints directly
(smooth, no IK during collection), while ``ee.*`` is added by forward kinematics.
With ``keep_joints=True`` the dataset stores the *superset* (``joint_*.pos`` AND
``ee.*``), so you can train a joint- or EE-space policy from the same data. IK is
only needed later, at EE-policy inference.

Run:  python examples/record_ee.py
"""

from lerobot_robot_trlc_dk1.follower import DK1FollowerConfig
from lerobot_robot_trlc_dk1.leader import DK1LeaderConfig
from lerobot_robot_trlc_dk1.ee_processors import make_dk1_processors

from lerobot.scripts.lerobot_record import record, RecordConfig, DatasetRecordConfig
from lerobot.configs.video import RGBEncoderConfig

# --- Cameras (strongly recommended for real policies; empty = state-only) --------
from lerobot.cameras.opencv import OpenCVCameraConfig
cameras = {
    "wrist": OpenCVCameraConfig(index_or_path=0, width=640, height=360, fps=60),
    "front": OpenCVCameraConfig(index_or_path=1, width=640, height=360, fps=60),
}

# --- Robot / teleop ---------------------------------------------------------------
robot_config = DK1FollowerConfig(
    port="/dev/tty.usbmodem00000000050C1",
    control_mode="impedance",
    cameras=cameras,
)
teleop_config = DK1LeaderConfig(
    port="/dev/tty.usbmodemE072A1F88AAC1",
)

# --- Video encoding ---------------------------------------------------------------
# https://huggingface.co/docs/lerobot/streaming_video_encoding
# Streaming encoding encodes frames to MP4 *during* capture (no PNG round-trip), so
# save_episode() is near-instant. Cost: the encoder shares CPU with the control loop.
# This setup is 2 cams x 640x360 @ 60 fps (~166M px/s) — mid-range; defaults + a
# couple of encoder threads are usually fine on Apple Silicon.
rgb_encoder = RGBEncoderConfig(
    #vcodec="libsvtav1",   # best compression + training perf, but most CPU-heavy.
    # If the CPU saturates (dropped-frame warnings / choppy robot / rerun lag),
    # switch to the Apple-Silicon hardware encoder — much lower CPU, larger files:
    vcodec="h264_videotoolbox",
    # pix_fmt="yuv420p",
    # g=2,                  # keyframe every 2 frames — fast random seek for training
    # crf=30,               # quality; lower = better/larger
)

# --- Dataset ----------------------------------------------------------------------
dataset_config = DatasetRecordConfig(
    repo_id="local/dk1_ee_pick2",     # namespace/name
    single_task="Pick up the object and place it in the bin.",
    fps=60,
    num_episodes=10,
    episode_time_s=30,
    reset_time_s=10,
    video=True,
    push_to_hub=False,               # set True (and `huggingface-cli login`) to upload
    # root="datasets/dk1_ee_pick",   # local path; defaults to ~/.cache/huggingface/lerobot
    # --- streaming video encoding ---
    streaming_encoding=True,         # encode live during capture (0.5.1 defaults to False)
    rgb_encoder=rgb_encoder,
    encoder_threads=2,               # threads per camera encoder; raise if CPU spare & frames drop, lower if starved
    encoder_queue_maxsize=60,        # buffered frames/cam (~1s at 60fps); overflow => dropped frames (warned)
)

record_config = RecordConfig(
    robot=robot_config,
    teleop=teleop_config,
    dataset=dataset_config,
    # display_data=True,               # live camera/state view (rerun)
)

# --- EE action representation (the only thing that makes this a Cartesian dataset) -
teleop_action_processor, robot_action_processor, robot_observation_processor = make_dk1_processors(
    action_space="ee",
    teleop_source="joint_leader",
    keep_joints=True,
    drive_via_ik=True,
)

if __name__ == "__main__":
    record(
        record_config,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )
