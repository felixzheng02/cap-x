"""Client-side tests for the split-process robot interface.

    cd cap-x
    .venv-client/bin/python -m pytest tests/test_remote_robot.py -v

The isolation tests always run. The rest need a simulator listening on CAPX_SIM_ENDPOINT and
skip cleanly otherwise:

    robocasa/capx_sim_server/.venv/bin/python robocasa/capx_sim_server/server.py --task OpenDrawer
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

ENDPOINT = os.environ.get("CAPX_SIM_ENDPOINT", "ipc:///tmp/capx-sim-0.sock")


# --------------------------------------------------------------- isolation

@pytest.mark.parametrize("module", ["mujoco", "robosuite", "robocasa", "torch"])
def test_simulator_packages_are_absent(module):
    """The boundary is structural, not a promise.

    If any of these become importable here, the harness has quietly re-acquired the ability to
    reach into the simulator in-process and the sim/real split has rotted.
    """
    assert importlib.util.find_spec(module) is None, (
        f"{module!r} is importable in the CaP-X client environment; the simulator must stay "
        "in its own process and environment"
    )


def test_robot_client_module_imports_without_a_simulator():
    from robot_driver import MotionProfile, RobotClient  # noqa: F401

    assert MotionProfile().min_base_cmd > 0.25, (
        "min_base_cmd must exceed the base stiction floor (frictionloss 250 vs gain 1000)"
    )


def test_remote_env_is_registered():
    from capx.envs.base import list_envs
    from capx.envs.tasks import list_configs
    from capx.integrations import list_apis

    assert "remote_robot_low_level" in list_envs()
    assert "remote_robot_code_env" in list_configs()
    assert "RemoteRobotApi" in list_apis()


def test_api_exposes_no_privileged_primitives():
    """Pure perception: ground-truth object queries have no real-robot equivalent and must not
    be reachable from generated code."""
    from capx.integrations.remote.control import RemoteRobotApi

    exposed = set(RemoteRobotApi.functions(RemoteRobotApi.__new__(RemoteRobotApi)))
    assert exposed == {
        "get_task_instruction", "get_observation", "segment", "get_depth",
        "mask_to_points", "goto_pose", "open_gripper", "close_gripper", "move_base",
    }
    for forbidden in ("get_object_pose", "list_objects", "object_registry", "visible_objects"):
        assert forbidden not in exposed, f"{forbidden} leaks privileged simulator state"
        assert not hasattr(RemoteRobotApi, forbidden)


# ------------------------------------------------------- live simulator ---

def _sim_available() -> bool:
    """Probe the simulator, don't just look for the socket FILE.

    A killed server leaves its ipc:// socket behind, so os.path.exists() reports True for a
    dead simulator and every live test then hangs until its timeout instead of skipping.
    """
    if ENDPOINT.startswith("ipc://") and not os.path.exists(ENDPOINT[len("ipc://"):]):
        return False
    try:
        import zmq

        from robot_driver import protocol as P

        ctx = zmq.Context()
        sock = ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.LINGER, 0)
        try:
            sock.connect(ENDPOINT)
            sock.send(P.pack({"id": 0, "method": P.PING, "params": {}}))
            return bool(sock.poll(1500))
        finally:
            sock.close(linger=0)
            ctx.term()
    except Exception:
        return False


live = pytest.mark.skipif(not _sim_available(), reason="no simulator listening")


@pytest.fixture(scope="module")
def rc():
    from robot_driver import RobotClient

    client = RobotClient(ENDPOINT)
    client.reset()
    yield client


@live
def test_spec_round_trip(rc):
    assert rc.action_dim == 13
    assert rc.base_mode_index == rc.action_dim - 1
    assert {"right", "right_gripper", "base"} <= set(rc.action_split)


@live
def test_reset_returns_episode_metadata(rc):
    meta = rc.reset()
    assert isinstance(meta["instruction"], str) and meta["instruction"]
    assert meta["rgb"].shape == (512, 512, 3)


@live
def test_move_to_joints_converges(rc):
    rc.reset()
    target = rc.joint_pos().copy()
    target[1] += 0.15
    assert rc.move_to_joints(target)
    assert np.linalg.norm(rc.joint_pos() - target) < 0.05


@live
def test_gripper_closes(rc):
    rc.reset()
    before = float(rc.last["gripper_qpos"][0])
    rc.close_gripper(settle_steps=25)
    assert float(rc.last["gripper_qpos"][0]) < before


@live
def test_drive_base_lateral_converges(rc):
    rc.reset()
    before = rc.base_pos()[:2].copy()
    assert rc.drive_base(0.0, 0.12, 0.0)
    assert float(np.linalg.norm(rc.base_pos()[:2] - before)) > 0.08


@live
def test_depth_has_calibration(rc):
    d = rc.get_depth()
    assert d["depth"].shape == (512, 512, 1)
    assert 0.05 < float(d["depth"].min()) < 5.0
    assert np.asarray(d["intrinsics"]).shape == (3, 3)
    assert np.asarray(d["cam_pose_mat"]).shape == (4, 4)


@live
def test_timeout_does_not_poison_the_socket(rc):
    """A DEALER socket must survive a timeout; REQ/REP would be unusable afterwards, and the
    harness retries timed-out trials up to three times."""
    from robot_driver import protocol as P

    with pytest.raises(TimeoutError):
        rc.call(P.STEP, {"action": rc.build_action().tolist()}, timeout_ms=1)
    # Socket still works for subsequent calls.
    assert rc.call(P.PING)["pong"] is True


@live
def test_video_frames_come_via_shared_memory(rc):
    rc.reset()
    rc.video_enable(True, clear=True)
    for _ in range(15):
        rc.step(rc.build_action())
    n = rc.video_frame_count()
    assert n >= 2
    frames = rc.video_frames_range(0, n)
    assert len(frames) == n and frames[0].shape == (512, 512, 3)


@live
def test_proxy_env_rejects_simulator_internals():
    from capx.envs.simulators.remote_robot import RemoteRobotEnv

    env = RemoteRobotEnv(endpoint=ENDPOINT)
    env.reset()
    with pytest.raises(AttributeError, match="no simulator internals|has no attribute"):
        _ = env.robosuite_env


@live
def test_proxy_video_signature_is_introspectable():
    """capx/envs/tasks/base.py inspects this signature to decide whether to pass wrist_camera;
    a *args/**kwargs forwarder silently breaks video capture."""
    import inspect

    from capx.envs.simulators.remote_robot import RemoteRobotEnv

    params = inspect.signature(RemoteRobotEnv.enable_video_capture).parameters
    assert "wrist_camera" in params and "clear" in params
