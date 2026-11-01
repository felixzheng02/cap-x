"""Compatibility shims for BEHAVIOR-1K v3.9.3.

CaP-X's BEHAVIOR integration was written against qingh097/b1k (BEHAVIOR v3.7.2), a fork that
patched OmniGibson in ways stock v3.9.3 does not. This module restores those behaviours on the
cap-x side so the call sites in r1pro_b1k.py stay unchanged.
"""

from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
)


class ArmSwitchablePrimitives(StarterSemanticActionPrimitives):
    """StarterSemanticActionPrimitives with a settable arm.

    cap-x's BEHAVIOR fork commented out the read-only ``arm`` property and added
    ``overwrite_arm(idx)``, which every two-armed R1Pro routine depends on. Stock v3.9.3 exposes
    ``arm`` as a read-only property returning ``robot.default_arm``, so arm switching is
    reinstated here rather than by patching BEHAVIOR.
    """

    _arm_override = None

    @property
    def arm(self):
        assert self.robot.is_manipulation, "Cannot use arm for non-manipulation robot"
        return self._arm_override if self._arm_override is not None else self.robot.default_arm

    def overwrite_arm(self, arm) -> None:
        """Select which arm subsequent primitives act on, by index into ``robot.arm_names``."""
        self._arm_override = self.robot.arm_names[arm]
