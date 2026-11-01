"""Ground-truth perception behind R1ProControlApi's exact interface.

Tier 1 of the BEHAVIOR baseline uses simulator ground truth rather than SAM3, so that a zero
score measures the agent's reasoning rather than a detector. The function names, signatures and
docstrings are inherited unchanged from :class:`R1ProControlApi`, which matters because the
prompt is generated from them by ``ApiBase.combined_doc()`` — only the source of the detections
differs. Swapping back to real perception is a matter of registering the parent class instead.

This also removes the SAM3 (8114), ContactGraspNet (8115) and Molmo server dependencies, so no
api_servers block is needed for a privileged run.
"""

import contextlib

import numpy as np
import open3d as o3d
import viser.transforms as vtf

from capx.integrations.r1pro import control as _control
from capx.integrations.r1pro.control import R1ProControlApi

# Server constructors the parent calls unconditionally in __init__.
_SERVER_INITS = (
    "init_contact_graspnet", "init_molmo", "init_sam3", "init_sam3_point_prompt",
    "init_owlvit", "init_sam2",
)


@contextlib.contextmanager
def _without_perception_servers():
    """Stub out the perception-server constructors for the duration of the parent's __init__.

    Preferred over reimplementing __init__: everything else it sets up (the pyroki context, the
    joint mappings, thresholds) stays exactly as upstream wrote it.
    """
    saved = {}
    for name in _SERVER_INITS:
        if hasattr(_control, name):
            saved[name] = getattr(_control, name)

            def _stub(_name=name):
                # The parent calls these factories in __init__ and stores what they return, so the
                # stub must itself be callable and hand back a placeholder that only fails if the
                # returned function is actually used.
                def _factory(*args, **kwargs):
                    def _unavailable(*a, **k):
                        raise RuntimeError(f"{_name} is unavailable in the privileged API")
                    return _unavailable
                return _factory

            setattr(_control, name, _stub())
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(_control, name, fn)


def _np(x):
    """Torch tensor or array-like -> float64 numpy array."""
    return np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x, dtype=np.float64)


class R1ProControlPrivilegedApi(R1ProControlApi):
    """R1ProControlApi with simulator ground truth in place of SAM3 / ContactGraspNet / Molmo."""

    def __init__(self, env, debug: bool = False, **kwargs) -> None:
        with _without_perception_servers():
            super().__init__(env, use_sam3=True, debug=debug, **kwargs)

    # ------------------------------------------------------------------ helpers
    def _resolve_object(self, object_name: str):
        """Resolve a natural-language name against the task object scope.

        Mirrors R1ProBehaviourLowLevel.get_object_state's substring matching, then falls back to
        matching any token, since the model writes names like "red radio" for "radio.n.01_1".
        """
        scope = self._env.task_relevant_obj
        needle = object_name.strip().lower()
        for candidate in (needle, needle.replace(" ", "_")):
            for key, value in scope.items():
                if value is not None and candidate in key.lower():
                    return value
        for token in sorted((t for t in needle.replace("_", " ").split() if len(t) > 2), key=len, reverse=True):
            for key, value in scope.items():
                if value is not None and token in key.lower():
                    return value
        # Same message the SAM3-backed parent raises, so the model sees identical feedback.
        raise ValueError("No sam3 detections")

    def _object_obb(self, obj):
        """Ground-truth oriented bounding box as an Open3D OBB, matching the parent's return type."""
        center, quat_xyzw, extent, _ = obj.get_base_aligned_bbox(visual=False, xy_aligned=False)
        center, quat_xyzw, extent = _np(center), _np(quat_xyzw), _np(extent)
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        rotation = vtf.SO3(wxyz=quat_wxyz).as_matrix()
        return o3d.geometry.OrientedBoundingBox(center, rotation, extent), center, quat_wxyz, extent

    # ------------------------------------------------------------------ perception
    def get_object_pose(self, object_name: str, return_bbox_extent: bool = False):
        obj = self._resolve_object(object_name)
        obb, center, quat_wxyz, extent = self._object_obb(obj)
        points = np.asarray(obb.get_box_points())
        print(f"Object position for {object_name}: {center}")
        print(f"Object quaternion wxyz for {object_name}: {quat_wxyz}")
        if return_bbox_extent:
            print(f"Object extent for {object_name}: {extent}")
            return center, quat_wxyz, extent, points, obb
        return center, quat_wxyz, None, points, obb

    def sample_grasp_pose(self, object_name: str):
        obj = self._resolve_object(object_name)
        obb, _, _, _ = self._object_obb(obj)
        # Upstream passes the natural-language name straight into _sample_grasp_pose, which looks
        # it up in the scene registry; ground truth gives us the registry name directly.
        pregrasp_pose, grasp_pose = self._env._sample_grasp_pose(obj.name, obb)
        return [pregrasp_pose], [grasp_pose]

    def detect_object_sam3(self, object_name: str, *args, **kwargs) -> bool:
        try:
            self._resolve_object(object_name)
            return True
        except ValueError:
            return False

    def find_object_base_rotate(self, object_name: str) -> None:
        # Ground truth needs no active search: the object is either in the task scope or absent.
        return self.detect_object_sam3(object_name)

    def find_object_torso_rotate(self, object_name: str) -> None:
        return self.detect_object_sam3(object_name)
