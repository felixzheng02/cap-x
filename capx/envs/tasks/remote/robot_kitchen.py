"""Code-execution env for a remote robot in a kitchen (simulated or real)."""

from capx.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling a mobile manipulator: a 7-DOF Franka Panda arm on a holonomic mobile base,
in a kitchen. You act by writing Python code that calls the APIs listed below.

Write ONE COMPLETE PROGRAM that attempts the whole task from start to finish. Do not write a
single exploratory call and stop -- your code is executed as a script, and you will NOT see any
intermediate output while it runs. Anything you print is shown to you only AFTER the whole
program has finished, and you may then be asked to revise it.

Because you cannot inspect results mid-script, write code that *branches on what it finds* and
prints what it saw, for example:

    instruction = get_task_instruction()
    print("task:", instruction)

    dets = segment("the drawer")
    print("drawer detections:", len(dets))
    if not dets:
        print("not visible; repositioning")
        move_base(0.0, 0.2, 0.0)
        dets = segment("the drawer")
    if dets:
        pts = mask_to_points(dets[0]["mask"])
        target = np.median(pts, axis=0)
        print("target:", target)
        open_gripper()
        goto_pose(target + np.array([0.0, 0.0, 0.10]), np.array([0.0, 1.0, 0.0, 0.0]))

Write ONLY executable Python code. Do not wrap it in code fences. Comments are fine. The API
functions are available as plain globals; import numpy explicitly if you need it.

How perception works:
- segment("<description>") is the only way to find things. An empty list means it is not
  visible from where the robot stands -- move_base(...) and look again.
- mask_to_points(mask) turns a detection into 3D points in the robot base frame; take their
  median or centroid as a target.
- goto_pose(position, quaternion_wxyz, z_approach=...) moves the arm only. It reaches about
  0.85 m from its base, so move_base closer before reaching for something far away.

Positions are metres, quaternions are WXYZ. Everything is in the CURRENT robot base frame, and
move_base changes that frame -- re-measure after moving.
"""

# Drives the whole harness offline (launch.py --use-oracle-code True) without an LLM. It is NOT
# a task solution: the agreed acceptance criterion is that the code runs, not that it succeeds.
ORACLE_CODE = """
import numpy as np

print("instruction:", get_task_instruction())

obs = get_observation()
print("rgb:", obs["robot0_robotview"]["images"]["rgb"].shape)
print("eef (base frame):", np.round(obs["robot_cartesian_pos"][:3], 3))

target_pts = None
for query in ["counter", "cabinet", "drawer", "fridge"]:
    dets = segment(query)
    print("segment(%r) -> %d detection(s)" % (query, len(dets)))
    if dets:
        pts = mask_to_points(dets[0]["mask"])
        print("   %s: %d points, median %s" % (dets[0]["label"], len(pts), np.round(np.median(pts, axis=0), 3)))
        if target_pts is None and len(pts):
            target_pts = pts

if target_pts is not None:
    centre = np.median(target_pts, axis=0)
    open_gripper()
    print("goto_pose ->", goto_pose(centre + np.array([0.0, 0.0, 0.15]), np.array([0.0, 1.0, 0.0, 0.0])))
    close_gripper()

print("move_base ->", move_base(0.0, 0.10, 0.0))
dets = segment("counter")
print("after move, counter visible:", len(dets) > 0)

RESULT = "oracle sequence complete"
"""


class RemoteRobotKitchenCodeEnv(CodeExecutionEnvBase):
    prompt = PROMPT
    oracle_code = ORACLE_CODE


__all__ = ["RemoteRobotKitchenCodeEnv"]
