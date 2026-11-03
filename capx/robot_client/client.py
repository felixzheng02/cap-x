"""Robot client: the layer that is identical for simulation and real hardware.

The boundary below this file is a raw ``step(action)`` / observation loop -- the same shape a
real robot driver exposes. Everything above it (closed-loop joint moves, gripper settling, base
motion, IK, Cartesian goals) lives **here**, client-side, so swapping the simulator for a Franka
means writing a new driver, not re-implementing the primitives.

Nothing in this module imports robosuite, mujoco or robocasa, by design.

Physics-specific constants live in :class:`MotionProfile`, not in the control logic, so that a
real robot can supply its own without editing shared code.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import protocol as P


@dataclass
class MotionProfile:
    """Tuning that depends on the *physical* robot, not on the control strategy."""

    #: Base joints in RoboCasa carry frictionloss=250 against an actuator gain of 1000, so any
    #: command below ~0.25 produces less force than stiction and the base does not move at all.
    #: A plain proportional law therefore stalls as the error shrinks and never starts short
    #: moves. Each axis is driven at no less than this until it is inside tolerance.
    min_base_cmd: float = 0.45
    base_gain: float = 4.0
    base_pos_tol: float = 0.02
    base_yaw_tol: float = 0.03
    base_max_steps: int = 1200

    joint_tol: float = 0.02
    joint_max_steps: int = 100
    gripper_settle_steps: int = 30

    #: panda_hand -> fingertip offset, applied in the end-effector frame.
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, -0.107)


SIM_PROFILE = MotionProfile()


class RobotClient:
    """Speaks the robot-driver protocol over ZeroMQ and provides closed-loop primitives."""

    def __init__(
        self,
        endpoint: str = P.DEFAULT_ENDPOINT,
        *,
        timeout_ms: int = 120_000,
        profile: MotionProfile | None = None,
    ) -> None:
        import zmq

        self._endpoint = endpoint
        self._timeout_ms = timeout_ms
        self.profile = profile or SIM_PROFILE

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.DEALER)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(endpoint)
        self._ids = itertools.count(1)

        # Static spec: fetched ONCE at connect. It cannot change without restarting the sim.
        # Per-episode data (instruction, max_steps) comes back from reset() instead.
        self.spec: dict[str, Any] = self.call(P.GET_SPEC)
        self.action_dim: int = int(self.spec["action_dim"])
        self.action_split: dict[str, list[int]] = self.spec["action_split"]
        self.base_mode_index: int = int(self.spec["base_mode_index"])

        self.last: dict[str, Any] = {}
        self._gripper = 1.0            # 1.0 open, 0.0 closed
        self._torso_target = 0.0
        self._ik = None
        self._cfg = None

    # ------------------------------------------------------------- transport

    def call(self, method: str, params: dict | None = None, timeout_ms: int | None = None):
        """One request/response round-trip.

        DEALER rather than REQ: a timeout here must not leave the socket unusable, because the
        CaP-X harness enforces a trial timeout with SIGALRM (which cannot reach the simulator
        process) and retries, so timeouts are routine.
        """
        rid = next(self._ids)
        self._sock.send(P.pack({"id": rid, "method": method, "params": params or {}}))
        deadline = timeout_ms if timeout_ms is not None else self._timeout_ms
        while True:
            if not self._sock.poll(deadline):
                raise TimeoutError(f"{method!r} timed out after {deadline} ms")
            resp = P.unpack(self._sock.recv())
            if resp.get("id") != rid:
                continue            # stale reply from an abandoned call; skip it
            if not resp.get("ok"):
                raise RuntimeError(f"simulator error in {method!r}: {resp.get('error')}")
            return resp.get("result")

    def close(self) -> None:
        try:
            self.call(P.CLOSE, timeout_ms=5000)
        except Exception:
            pass
        self._sock.close(linger=0)

    # ------------------------------------------------------ driver-level API

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self.last = self.call(P.RESET, {"seed": seed})
        self._gripper = 1.0
        self._cfg = None
        return self.last

    def step(self, action: np.ndarray) -> dict[str, Any]:
        self.last = self.call(P.STEP, {"action": np.asarray(action, float).tolist()})
        return self.last

    def get_depth(self, camera: str | None = None) -> dict[str, Any]:
        return self.call(P.GET_DEPTH, {"camera": camera})

    def get_segmentation(self, camera: str | None = None) -> dict[str, Any]:
        return self.call(P.GET_SEGMENTATION, {"camera": camera})

    def render(self, camera: str | None = None) -> np.ndarray:
        return self.call(P.RENDER, {"camera": camera})["rgb"]

    def evaluate(self) -> dict[str, Any]:
        """Sim-only scoring channel. Never exposed to generated code."""
        return self.call(P.EVALUATE)

    # --------------------------------------------------------- action vector

    def build_action(
        self,
        joints: np.ndarray | None = None,
        *,
        base_vel: np.ndarray | None = None,
    ) -> np.ndarray:
        """Assemble the driver's action vector from the layout reported by ``get_spec``.

        The part order is NOT stable across environments (RoboCasa and robosuite's own ``Lift``
        differ), which is exactly why it is queried rather than hardcoded.
        """
        a = np.zeros(self.action_dim, dtype=np.float64)
        target = self.joint_pos() if joints is None else np.asarray(joints, float)

        s, e = self.action_split["right"]
        a[s:e] = np.asarray(target, float).reshape(-1)[: e - s]
        if "torso" in self.action_split:
            s, e = self.action_split["torso"]
            a[s:e] = self._torso_target
        if base_vel is not None and "base" in self.action_split:
            s, e = self.action_split["base"]
            a[s:e] = np.asarray(base_vel, float).reshape(-1)[: e - s]
        # GRIP convention: -1 fully open, +1 fully closed.
        s, e = self.action_split["right_gripper"]
        a[s:e] = 1.0 - self._gripper * 2.0
        # base_mode > 0 drives the base and makes the arm track its desired pose.
        a[self.base_mode_index] = 1.0 if base_vel is not None else -1.0
        return a

    # --------------------------------------------------------------- getters

    def joint_pos(self) -> np.ndarray:
        return np.asarray(self.last["joint_pos"], dtype=np.float64)

    def base_pos(self) -> np.ndarray:
        return np.asarray(self.last["base_pos"], dtype=np.float64)

    def base_yaw(self) -> float:
        import viser.transforms as vtf

        return float(
            vtf.SO3(wxyz=np.asarray(self.last["base_quat_wxyz"], float)).as_rpy_radians().yaw
        )

    def sim_step_count(self) -> int:
        return int(self.last.get("sim_step_count", 0))

    # ---------------------------------------------------- closed-loop moves

    def move_to_joints(self, joints, *, tolerance=None, max_steps=None) -> bool:
        """Drive the arm to absolute joint targets. Returns True on convergence."""
        tol = self.profile.joint_tol if tolerance is None else tolerance
        steps = self.profile.joint_max_steps if max_steps is None else max_steps
        target = np.asarray(joints, float).reshape(-1)[:7]
        for _ in range(steps):
            if np.linalg.norm(self.joint_pos() - target) < tol:
                return True
            self.step(self.build_action(target))
        return np.linalg.norm(self.joint_pos() - target) < tol

    def set_gripper(self, fraction: float, *, settle_steps: int | None = None) -> None:
        self._gripper = float(np.clip(fraction, 0.0, 1.0))
        for _ in range(self.profile.gripper_settle_steps if settle_steps is None else settle_steps):
            self.step(self.build_action())

    def open_gripper(self, settle_steps: int | None = None) -> None:
        self.set_gripper(1.0, settle_steps=settle_steps)

    def close_gripper(self, settle_steps: int | None = None) -> None:
        self.set_gripper(0.0, settle_steps=settle_steps)

    def drive_base(self, dx: float, dy: float, dyaw: float) -> bool:
        """Relative base motion in the current base frame. Returns True on convergence.

        The base is velocity-controlled, so this is a closed loop. There is no collision
        checking: a blocked path simply fails to converge and returns False.
        """
        pr = self.profile
        start, yaw0 = self.base_pos()[:2], self.base_yaw()
        c, s = np.cos(yaw0), np.sin(yaw0)
        goal = start + np.array([c * dx - s * dy, s * dx + c * dy])
        goal_yaw = yaw0 + dyaw

        for _ in range(pr.base_max_steps):
            cur, yaw = self.base_pos()[:2], self.base_yaw()
            err_w = goal - cur
            err_yaw = float(np.arctan2(np.sin(goal_yaw - yaw), np.cos(goal_yaw - yaw)))
            if np.linalg.norm(err_w) < pr.base_pos_tol and abs(err_yaw) < pr.base_yaw_tol:
                self.step(self.build_action(base_vel=np.zeros(3)))
                return True
            cc, ss = np.cos(yaw), np.sin(yaw)
            err_l = np.array([cc * err_w[0] + ss * err_w[1], -ss * err_w[0] + cc * err_w[1]])
            vel = np.zeros(3)
            for i, (err, tol) in enumerate(
                zip((err_l[0], err_l[1], err_yaw), (pr.base_pos_tol, pr.base_pos_tol, pr.base_yaw_tol))
            ):
                if abs(err) > tol:
                    vel[i] = np.sign(err) * min(1.0, max(pr.min_base_cmd, pr.base_gain * abs(err)))
            self.step(self.build_action(base_vel=vel))
        return False

    # ----------------------------------------------------------- Cartesian

    @property
    def ik(self):
        if self._ik is None:
            from capx.integrations.motion.pyroki import init_pyroki

            self._ik = init_pyroki()
        return self._ik

    def goto_pose(self, position, quaternion_wxyz, z_approach: float = 0.0) -> dict[str, Any]:
        """Cartesian goal in the robot base frame, via IK and joint-space waypoints."""
        from scipy.spatial.transform import Rotation as SciRotation

        pos = np.asarray(position, float).reshape(3)
        quat = np.asarray(quaternion_wxyz, float).reshape(4)
        quat = quat / max(np.linalg.norm(quat), 1e-9)
        rot = SciRotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        target = pos + rot.apply(np.asarray(self.profile.tcp_offset, float))

        waypoints = []
        if z_approach:
            waypoints.append(target + rot.apply(np.array([0.0, 0.0, -z_approach])))
        waypoints.append(target)

        ok = True
        for wp in waypoints:
            self._cfg = self.ik(
                target_pose_wxyz_xyz=np.concatenate([quat, wp]), prev_cfg=self._cfg
            )
            ok = self.move_to_joints(np.asarray(self._cfg)[:7]) and ok
        return {"ok": bool(ok), "joints": self.joint_pos().tolist()}

    # ------------------------------------------------------------ video ---

    def video_enable(self, enabled=True, clear=True, wrist=False) -> None:
        self.call(P.VIDEO_ENABLE, {"enabled": enabled, "clear": clear, "wrist": wrist})

    def video_frame_count(self) -> int:
        return int(self.call(P.VIDEO_COUNT)["count"])

    def video_frames_range(self, start: int, end: int, *, wrist: bool = False) -> list[np.ndarray]:
        """Pull frames via the filesystem (/dev/shm), never over the wire.

        A trial's buffer is hundreds of MB; serialising it through the socket is the single
        worst thing that could be put on this boundary.
        """
        import os
        import tempfile

        if end <= start:
            return []
        fd, path = tempfile.mkstemp(suffix=".npy", dir="/dev/shm")
        os.close(fd)
        try:
            self.call(P.VIDEO_DUMP, {"start": start, "end": end, "path": path, "wrist": wrist})
            return list(np.load(path))
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
