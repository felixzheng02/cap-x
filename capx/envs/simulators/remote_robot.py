"""BaseEnv proxy over a remote robot driver (simulated or real).

A thin shell over :class:`robot_driver.RobotClient`. It exists only because the CaP-X
harness needs a ``BaseEnv``; all real behaviour lives in the client, which is the piece shared
with real hardware.

Deliberately **explicit**: there is no catch-all ``__getattr__``. CaP-X injects the low-level env
into the LLM's exec namespace as the global ``env``, and in the in-process backends that let
generated code reach straight into ``robosuite_env.sim``. Across a process boundary that is
impossible anyway, and a privileged shortcut into simulator state is exactly what makes sim
results fail to transfer -- so unknown attributes raise a message pointing at the API instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from robot_driver import RobotClient


def _init_pyroki():
    """Injected into RobotClient so the shared driver imports no ``capx`` name.

    Imported lazily: it reaches a pyroki service, and a client that never makes a
    Cartesian move should not pay for it at construction.
    """
    from capx.integrations.motion.pyroki import init_pyroki

    return init_pyroki()


class RemoteRobotEnv(BaseEnv):
    """Harness-facing proxy. Owns no simulator state of its own."""

    def __init__(
        self,
        *,
        privileged: bool = False,
        enable_render: bool = False,
        viser_debug: bool = False,
        endpoint: str | None = None,
        timeout_ms: int = 120_000,
    ) -> None:
        super().__init__()
        from robot_driver import protocol as P

        self.privileged = privileged
        self.enable_render = enable_render
        self.viser_debug = viser_debug

        self.client = RobotClient(
            endpoint or P.DEFAULT_ENDPOINT,
            timeout_ms=timeout_ms,
            ik_factory=_init_pyroki,
        )
        self.max_steps = int(self.client.spec.get("max_steps", 5000))
        self._instruction = ""
        self._wrist_recording = False

    # ------------------------------------------------------------ gym core

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        meta = self.client.reset(seed=seed)
        self._instruction = meta.get("instruction", "")
        self.max_steps = int(meta.get("max_steps", self.max_steps))
        return self.get_observation(), {"task_prompt": self._instruction}

    def step(self, action: Any):
        obs = self.get_observation()
        return obs, self.compute_reward(), False, self._sim_step_count >= self.max_steps, {}

    def get_observation(self) -> dict[str, Any]:
        """RGB + robot state. Depth and segmentation are fetched explicitly by the API.

        The scene RGB is published under ``robot0_robotview`` because that is the key CaP-X's
        utilities look for; it carries no implication about which physical camera it is.
        """
        last = self.client.last or self.client.reset()
        rgb = last.get("rgb")
        obs: dict[str, Any] = {
            "robot0_robotview": {"images": {"rgb": rgb}},
            "robot_joint_pos": np.concatenate(
                [np.asarray(last["joint_pos"], np.float32),
                 [float(last["gripper_qpos"][0]) / 0.04]]
            ),
            "robot_cartesian_pos": self._eef_in_base(last),
            "instruction": self._instruction,
        }
        return obs

    def _eef_in_base(self, last: dict[str, Any]) -> np.ndarray:
        import viser.transforms as vtf

        base = vtf.SE3(wxyz_xyz=np.concatenate(
            [np.asarray(last["base_quat_wxyz"], float), np.asarray(last["base_pos"], float)]))
        eef = vtf.SE3(wxyz_xyz=np.concatenate(
            [np.asarray(last["eef_quat_wxyz"], float), np.asarray(last["eef_pos"], float)]))
        rel = base.inverse() @ eef
        return np.concatenate(
            [rel.translation(), rel.rotation().wxyz, [float(last["gripper_qpos"][0]) / 0.04]]
        )

    # ------------------------------------------- evaluation (harness only)

    def compute_reward(self) -> float:
        return float(self.client.evaluate()["reward"])

    def task_completed(self) -> bool:
        return bool(self.client.evaluate()["success"])

    def task_prompt(self) -> str:
        return self._instruction

    @property
    def _sim_step_count(self) -> int:
        return self.client.sim_step_count()

    # --------------------------------------------------------- rendering

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")
        return self.client.render()

    def render_wrist(self) -> np.ndarray:
        return self.client.render(self.client.spec["cameras"]["wrist"])

    # ------------------------------------------------------------- video

    # NOTE: the signature matters. capx/envs/tasks/base.py INTROSPECTS it with
    # inspect.signature(...).parameters to decide whether to pass wrist_camera, so a
    # *args/**kwargs forwarder silently breaks video capture.
    def enable_video_capture(
        self, enabled: bool = True, *, clear: bool = True, wrist_camera: bool = False
    ) -> None:
        self._wrist_recording = wrist_camera
        self.client.video_enable(enabled, clear=clear, wrist=wrist_camera)

    def get_video_frame_count(self) -> int:
        return self.client.video_frame_count()

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return self.client.video_frames_range(start, end)

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = self.client.video_frames_range(0, self.client.video_frame_count())
        if clear:
            self.client.video_enable(self._is_recording(), clear=True, wrist=self._wrist_recording)
        return frames

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        return self.client.video_frames_range(
            0, self.client.video_frame_count(), wrist=True
        ) if self._wrist_recording else []

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return self.client.video_frames_range(start, end, wrist=True) if self._wrist_recording else []

    def _is_recording(self) -> bool:
        return True

    def close(self) -> None:
        self.client.close()

    # ------------------------------------------------------------- guard

    def __getattr__(self, name: str):
        # Only called when normal lookup fails.
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(
            f"{type(self).__name__} has no attribute {name!r}. This environment talks to a "
            "robot over IPC and deliberately exposes no simulator internals -- use the API "
            "functions (segment, get_depth, goto_pose, move_base, ...) instead."
        )


__all__ = ["RemoteRobotEnv"]
