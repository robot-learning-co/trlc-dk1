"""A sync inference engine for relative-action policies.

A relative-action policy generates a chunk of actions relative to a single
reference state, captured in ``RelativeActionsProcessorStep._last_state``.
If that reference is refreshed every control tick, cached chunk actions
popped on later ticks get re-anchored to the *current* robot state, so the
resulting absolute targets drift through the chunk.

This engine avoids that drift by:

1. Calling ``policy.predict_action_chunk(batch)`` (not ``select_action``)
   when the local FIFO is empty.
2. Postprocessing **all** ``n_action_steps`` actions in the chunk in one
   go, while the relative step's state cache still reflects the
   chunk-generation observation. Every action in the chunk is therefore
   converted to absolute against the same reference state — exactly the
   UMI "relative trajectory" semantics that the model was trained for.
3. Serving those absolute actions one-per-tick from a local FIFO.
4. Still running the preprocessor and diffusion observation-queue update
   every tick so the *next* chunk-generation reads a fresh
   ``n_obs_steps`` window. The resulting per-tick state-cache updates
   are harmless: only generation ticks read the cache (through the
   postprocessor loop), and they read it immediately after preprocessing
   the current frame.

Importing this module is enough to (a) register
``--inference.type=sync_relative`` with draccus and (b) monkey-patch
``lerobot.rollout.inference.factory.create_inference_engine`` to know how
to build this engine.

Usage — import for the side-effect, then select it on the rollout CLI::

    from lerobot_robot_trlc_dk1 import sync_relative_inference  # noqa: F401

    # python examples/ee_rollout.py \\
    #     --policy.path=<relative-action checkpoint> \\
    #     --inference.type=sync_relative \\
    #     --device=cuda --fps=30 --duration=300 --task="..."

The engine requires a ``DiffusionPolicy``-style policy (one exposing
``_queues`` and ``predict_action_chunk``); other policy types are rejected
at construction.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    make_robot_action,
    populate_queues,
    prepare_observation_for_inference,
)
from lerobot.processor import PolicyProcessorPipeline
from lerobot.rollout.inference import factory as _engine_factory
from lerobot.rollout.inference.base import InferenceEngine
from lerobot.rollout.inference.factory import InferenceEngineConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES

logger = logging.getLogger(__name__)


@InferenceEngineConfig.register_subclass("sync_relative")
@dataclass
class SyncRelativeInferenceConfig(InferenceEngineConfig):
    """Inline synchronous inference for relative-action policies.

    Generates the full action chunk via ``predict_action_chunk``,
    postprocesses all actions against the chunk-time state, then serves
    them one-per-tick. Equivalent to UMI's relative-trajectory rollout.
    """


class SyncRelativeInferenceEngine(InferenceEngine):
    """Synchronous inference engine that is safe for relative-action policies."""

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        device: str | None,
        robot_type: str,
    ) -> None:
        # _update_policy_obs_queue mirrors DiffusionPolicy's queue plumbing
        # (`_queues` + `predict_action_chunk`); other policies (e.g. ACT with
        # its `_action_queue`) would fail silently mid-rollout, so reject them
        # up front.
        if not hasattr(policy, "_queues") or not callable(
            getattr(policy, "predict_action_chunk", None)
        ):
            raise TypeError(
                f"sync_relative requires a DiffusionPolicy-style policy with "
                f"`_queues` and `predict_action_chunk`; got {type(policy).__name__}."
            )
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type
        self._absolute_chunk: deque[torch.Tensor] = deque()
        logger.info(
            "SyncRelativeInferenceEngine initialized (device=%s, action_keys=%d, "
            "n_action_steps=%d)",
            self._device,
            len(ordered_action_keys),
            int(getattr(policy.config, "n_action_steps", 0)),
        )

    # ----- lifecycle ----------------------------------------------------
    def start(self) -> None:
        logger.info("SyncRelativeInferenceEngine started")

    def stop(self) -> None:
        logger.info("SyncRelativeInferenceEngine stopped")

    def reset(self) -> None:
        logger.info("Resetting sync_relative inference state (policy + processors + FIFO)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        self._absolute_chunk.clear()

    # ----- per-tick action production -----------------------------------
    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        if obs_frame is None:
            return None

        # Cheap replay: while the FIFO still holds more than the last
        # (n_obs_steps - 1) precomputed actions, serve them directly — no
        # preprocessing, no model, no queue update. Running the full image
        # preprocessor on *every* tick adds dispatch jitter (more so the more
        # cameras the robot has), so only the re-plan neighbourhood pays that
        # cost; serving stays fast and uniform between chunks. Served actions
        # are unchanged (precomputed at refill).
        n_obs = int(getattr(self._policy.config, "n_obs_steps", 1) or 1)
        if len(self._absolute_chunk) > max(n_obs - 1, 0):
            return self._pop_ordered()

        # Re-plan neighbourhood: refresh the policy's obs queue (and the
        # relative step's _last_state) so that at generation the n_obs_steps
        # history is made of consecutive control ticks, exactly as in training.
        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            observation = prepare_observation_for_inference(
                observation, self._device, self._task, self._robot_type
            )
            observation = self._preprocessor(observation)
            batch = self._update_policy_obs_queue(observation)

            if self._absolute_chunk:
                return self._pop_ordered()

            self._refill_chunk(batch)

        return self._pop_ordered()

    def _pop_ordered(self) -> torch.Tensor:
        absolute_action = self._absolute_chunk.popleft().squeeze(0).cpu()
        action_dict = make_robot_action(absolute_action, self._dataset_features)
        return torch.tensor([action_dict[k] for k in self._ordered_action_keys])

    # ----- helpers ------------------------------------------------------
    def _update_policy_obs_queue(self, observation: dict[str, Any]) -> dict[str, Any]:
        # Mirror ``DiffusionPolicy.select_action``'s queue plumbing: pop ACTION
        # from the input, stack camera streams into ``observation.images`` if
        # the policy uses separate encoders, then populate the policy's obs
        # queues so ``predict_action_chunk`` can stack history along dim=1.
        batch = dict(observation)
        if ACTION in batch:
            batch.pop(ACTION)
        if getattr(self._policy.config, "image_features", None):
            batch[OBS_IMAGES] = torch.stack(
                [batch[k] for k in self._policy.config.image_features], dim=-4
            )
        self._policy._queues = populate_queues(self._policy._queues, batch)
        return batch

    def _refill_chunk(self, batch: dict[str, Any]) -> None:
        """Generate a fresh chunk and store its absolute actions in the FIFO."""
        start = time.perf_counter()
        # Relative chunk shaped (B, n_action_steps, action_dim).
        action_chunk = self._policy.predict_action_chunk(batch)

        # Convert every action in the chunk to absolute using the SAME
        # state cache (the one captured by the preprocessor a moment ago).
        for t in range(action_chunk.shape[1]):
            relative_action = action_chunk[:, t, :]
            absolute_action = self._postprocessor(relative_action)
            self._absolute_chunk.append(absolute_action)

        logger.info(
            "sync_relative chunk refill: %d actions in %.2fs",
            action_chunk.shape[1],
            time.perf_counter() - start,
        )


# ---------------------------------------------------------------------------
# Wire this engine into ``create_inference_engine`` so build_rollout_context
# can instantiate it. ``create_inference_engine`` dispatches with a hard-coded
# if/elif over engine config classes, so we monkey-patch a shim that handles
# our subclass first and falls back to the original otherwise.
# ---------------------------------------------------------------------------

_original_create_inference_engine = _engine_factory.create_inference_engine


def _create_inference_engine_with_relative(
    config: InferenceEngineConfig,
    *,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    robot_wrapper,  # ThreadSafeRobot, untyped to avoid the import.
    hw_features: dict,
    dataset_features: dict,
    ordered_action_keys: list[str],
    task: str,
    fps: float,
    device: str | None,
    use_torch_compile: bool = False,
    compile_warmup_inferences: int = 2,
    shutdown_event=None,
):
    if isinstance(config, SyncRelativeInferenceConfig):
        logger.info("Creating inference engine: sync_relative")
        return SyncRelativeInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys,
            task=task,
            device=device,
            robot_type=robot_wrapper.robot_type,
        )
    return _original_create_inference_engine(
        config,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=robot_wrapper,
        hw_features=hw_features,
        dataset_features=dataset_features,
        ordered_action_keys=ordered_action_keys,
        task=task,
        fps=fps,
        device=device,
        use_torch_compile=use_torch_compile,
        compile_warmup_inferences=compile_warmup_inferences,
        shutdown_event=shutdown_event,
    )


_engine_factory.create_inference_engine = _create_inference_engine_with_relative
# Also patch the re-export used by ``build_rollout_context`` if it imported
# the symbol directly.
try:
    from lerobot.rollout import context as _rollout_context  # noqa: WPS433

    if hasattr(_rollout_context, "create_inference_engine"):
        _rollout_context.create_inference_engine = _create_inference_engine_with_relative
except ImportError:
    pass


__all__ = ["SyncRelativeInferenceConfig", "SyncRelativeInferenceEngine"]
