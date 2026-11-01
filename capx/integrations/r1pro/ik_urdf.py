"""Generate the 18-DoF R1Pro URDF that the pyroki IK layer expects.

cap-x loads ``models/r1pro/urdf/r1pro_ik.urdf`` from the robot-assets dataset. That file ships
with the BEHAVIOR fork cap-x was built against (qingh097/b1k) but **not** with BEHAVIOR-1K
v3.9.3, whose dataset only has the full 28-DoF ``r1pro.urdf``.

Substituting the 28-DoF URDF would be silently wrong: ``R1ProControlApi.pyroki_joint_mapping``
and ``controller_joint_mapping`` are hardcoded 18-element index permutations, so a 28-joint IK
solution would be scattered across the wrong controller joints.

The file is reconstructable exactly. Taking the stock ``r1pro.urdf`` and fixing the 6 base
wheel/steer joints and the 4 gripper finger joints leaves exactly 18 actuated joints, in
document order:

    torso_joint1..4, left_arm_joint1..7, right_arm_joint1..7

which is precisely the ordering both hardcoded mappings imply. The result is written inside the
cap-x tree with a ``meshes`` symlink back to the dataset, so the shared dataset stays untouched.
"""

import os
import xml.etree.ElementTree as ET

from omnigibson.utils.asset_utils import get_dataset_path

# Joints present in the full robot but absent from the IK chain.
_FIXED_JOINTS = (
    "steer_motor_joint1", "steer_motor_joint2", "steer_motor_joint3",
    "wheel_motor_joint1", "wheel_motor_joint2", "wheel_motor_joint3",
    "left_gripper_finger_joint1", "left_gripper_finger_joint2",
    "right_gripper_finger_joint1", "right_gripper_finger_joint2",
)

# The actuated joints the generated file must expose, in this exact order.
EXPECTED_IK_JOINTS = (
    ["torso_joint%d" % i for i in (1, 2, 3, 4)]
    + ["left_arm_joint%d" % i for i in range(1, 8)]
    + ["right_arm_joint%d" % i for i in range(1, 8)]
)

_ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "envs", "assets", "r1pro")


def ensure_r1pro_ik_urdf(force: bool = False) -> str:
    """Return the path to r1pro_ik.urdf, generating it from the stock URDF if needed."""
    out_path = os.path.join(_ASSET_DIR, "r1pro_ik.urdf")
    if os.path.exists(out_path) and not force:
        return out_path

    src_dir = os.path.join(get_dataset_path("omnigibson-robot-assets"), "models", "r1pro", "urdf")
    src_path = os.path.join(src_dir, "r1pro.urdf")
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Stock R1Pro URDF not found at {src_path}")

    tree = ET.parse(src_path)
    root = tree.getroot()
    fixed = set()
    for joint in root.findall("joint"):
        if joint.get("name") in _FIXED_JOINTS:
            joint.set("type", "fixed")
            # A fixed joint has no axis, limit or dynamics; drop them so strict parsers are happy.
            for tag in ("axis", "limit", "dynamics", "mimic", "safety_controller"):
                for child in joint.findall(tag):
                    joint.remove(child)
            fixed.add(joint.get("name"))

    missing = set(_FIXED_JOINTS) - fixed
    if missing:
        raise RuntimeError(f"Expected joints absent from {src_path}: {sorted(missing)}")

    actuated = [j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"]
    if actuated != EXPECTED_IK_JOINTS:
        raise RuntimeError(
            "Generated IK URDF joint order does not match the order cap-x's hardcoded "
            f"pyroki_joint_mapping assumes.\n  got:      {actuated}\n  expected: {EXPECTED_IK_JOINTS}"
        )

    os.makedirs(_ASSET_DIR, exist_ok=True)
    # Mesh references are relative ("meshes/..."), so they must resolve next to the URDF.
    link = os.path.join(_ASSET_DIR, "meshes")
    if not os.path.islink(link) and not os.path.exists(link):
        os.symlink(os.path.join(src_dir, "meshes"), link)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


if __name__ == "__main__":
    path = ensure_r1pro_ik_urdf(force=True)
    print(f"wrote {path} with {len(EXPECTED_IK_JOINTS)} actuated joints")
