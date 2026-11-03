"""Agent-facing primitives for a remote robot (simulated or real).

Docstrings here ARE the specification shown to the model: ``ApiBase.combined_doc()`` renders
every signature and docstring into the prompt.

**No privileged information.** There is no ``get_object_pose`` and no ``list_objects``: neither
has a real-robot equivalent. The agent perceives with ``segment()`` -- which goes through
CaP-X's ordinary SAM3 client and is answered by whatever is listening on port 8114 -- and
recovers 3D geometry from the mask, the depth image and the camera calibration, exactly as it
would on hardware.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from capx.integrations.base_api import ApiBase


class RemoteRobotApi(ApiBase):
    """Primitives for a Franka-style arm on a mobile base, driven over IPC."""

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self._segment_fn = None

    @property
    def _segment(self):
        if self._segment_fn is None:
            from capx.integrations.vision.sam3 import init_sam3

            self._segment_fn = init_sam3()
        return self._segment_fn

    @property
    def _client(self):
        return self._env.client

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {
            "get_task_instruction": self.get_task_instruction,
            "get_observation": self.get_observation,
            "segment": self.segment,
            "get_depth": self.get_depth,
            "mask_to_points": self.mask_to_points,
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "move_base": self.move_base,
        }

    # ------------------------------------------------------------ sensing

    def get_task_instruction(self) -> str:
        """Return the natural-language instruction for this episode.

        Returns:
            str: e.g. "Open the right drawer."
        """
        return self._env.task_prompt()

    def get_observation(self) -> dict[str, Any]:
        """Get the current observation.

        Returns:
            dict: obs["robot0_robotview"]["images"]["rgb"] is the (H, W, 3) uint8 camera view.
                obs["robot_cartesian_pos"] is the end-effector pose in the robot base frame as
                xyz + wxyz + normalised gripper opening.
        """
        return self._env.get_observation()

    def segment(self, text_prompt: str) -> list[dict[str, Any]]:
        """Segment the current camera view with a natural-language prompt.

        This is the only way to find things. Describe what you want in plain words
        ("the drawer", "a counter", "the fridge").

        Args:
            text_prompt: What to look for.

        Returns:
            list[dict]: one entry per detection, best first, each with
                mask ((H, W) bool), box ([x0, y0, x1, y1]), score (float), label (str).
                An empty list means it was not found in the current view -- try move_base to
                change viewpoint, then look again.
        """
        rgb = self._env.get_observation()["robot0_robotview"]["images"]["rgb"]
        results = self._segment(rgb, text_prompt) or []
        out = []
        for r in results:
            mask = r.get("mask")
            if mask is None:
                continue
            out.append(
                {
                    "mask": np.asarray(mask).astype(bool),
                    "box": r.get("box"),
                    "score": float(r.get("score", 0.0)),
                    "label": r.get("label", ""),
                }
            )
        self._log_step("segment", f"{text_prompt!r} -> {len(out)} detection(s)")
        return out

    def get_depth(self) -> dict[str, Any]:
        """Get the depth image and camera calibration for the current view.

        Returns:
            dict: depth ((H, W, 1) float32, metres), intrinsics ((3, 3)), and
                cam_pose_mat ((4, 4) camera pose in the robot base frame).
        """
        return self._client.get_depth()

    def mask_to_points(self, mask: np.ndarray, max_points: int = 4000) -> np.ndarray:
        """Back-project a segmentation mask into 3D points in the ROBOT BASE frame.

        Combines the current depth image with the camera calibration. Use the median or centroid
        of the returned points as a target for goto_pose.

        Args:
            mask: (H, W) boolean array, e.g. segment(...)[0]["mask"].
            max_points: Subsample if the mask covers more pixels than this.

        Returns:
            np.ndarray: (N, 3) points in metres, robot base frame. Empty if the mask has no
                valid depth.
        """
        d = self._client.get_depth()
        depth = np.asarray(d["depth"])[:, :, 0]
        K = np.asarray(d["intrinsics"], dtype=np.float64)
        cam = np.asarray(d["cam_pose_mat"], dtype=np.float64)

        m = np.asarray(mask).astype(bool) & (depth > 0)
        vs, us = np.nonzero(m)
        if vs.size == 0:
            return np.zeros((0, 3))
        if vs.size > max_points:
            sel = np.random.default_rng(0).choice(vs.size, max_points, replace=False)
            vs, us = vs[sel], us[sel]
        z = depth[vs, us]
        x = (us - K[0, 2]) * z / K[0, 0]
        y = (vs - K[1, 2]) * z / K[1, 1]
        pts_cam = np.stack([x, y, z], axis=1)
        return pts_cam @ cam[:3, :3].T + cam[:3, 3]

    # ------------------------------------------------------------ control

    def goto_pose(
        self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0
    ) -> dict[str, Any]:
        """Move the end-effector to a pose in the robot base frame, using inverse kinematics.

        The arm only; the base does not move. Targets beyond the arm's reach (~0.85 m from the
        base) will not be achieved -- use move_base to get closer first.

        Args:
            position: (3,) target XYZ in metres, robot base frame.
            quaternion_wxyz: (4,) target orientation, WXYZ.
            z_approach: If non-zero, first move this many metres back along the gripper approach
                axis, then descend. Useful for grasps.

        Returns:
            dict: {"ok": bool}
        """
        res = self._client.goto_pose(position, quaternion_wxyz, z_approach)
        self._log_step("goto_pose", f"-> {np.array2string(np.asarray(position), precision=3)} ok={res['ok']}")
        return {"ok": res["ok"]}

    def open_gripper(self) -> None:
        """Open the gripper."""
        self._client.open_gripper()

    def close_gripper(self) -> None:
        """Close the gripper to grasp whatever is between the fingers."""
        self._client.close_gripper()

    def move_base(self, dx: float, dy: float, dyaw: float) -> dict[str, Any]:
        """Drive the mobile base by a RELATIVE amount in the current base frame.

        +dx forward, +dy left, +dyaw counter-clockwise (radians).

        WARNING: this moves the frame that every pose is expressed in. Anything you measured
        before calling this is stale afterwards -- segment and re-measure after moving.
        There is no collision checking: a blocked path returns ok=False rather than raising.

        Args:
            dx: Forward/backward metres.
            dy: Left/right metres.
            dyaw: Rotation in radians.

        Returns:
            dict: {"ok": bool, "moved": float metres actually travelled}
        """
        before = self._client.base_pos()[:2].copy()
        ok = self._client.drive_base(float(dx), float(dy), float(dyaw))
        moved = float(np.linalg.norm(self._client.base_pos()[:2] - before))
        self._log_step("move_base", f"({dx:.2f},{dy:.2f},{dyaw:.2f}) moved={moved:.3f} ok={ok}")
        return {"ok": bool(ok), "moved": moved}


__all__ = ["RemoteRobotApi"]
