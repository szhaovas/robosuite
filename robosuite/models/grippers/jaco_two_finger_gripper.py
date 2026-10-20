"""
Tendon-coupled two-finger Jaco gripper.
"""

import numpy as np
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class JacoTwoFingerGripperBase(GripperModel):
    """
    Tendon-coupled two-finger Jaco gripper.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(
            xml_path_completion("grippers/jaco_two_finger_gripper.xml"), idn=idn
        )

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        # Fully open proximal joints and the measured settled distal angles.
        return np.array([0.0, 0.92637, 0.0, 0.92637])

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "index_proximal_collision",
                "index_distal_collision",
                "index_tip_collision",
            ],
            "right_finger": [
                "thumb_proximal_collision",
                "thumb_distal_collision",
                "thumb_tip_collision",
            ],
            "left_fingerpad": ["index_pad_collision"],
            "right_fingerpad": ["thumb_pad_collision"],
        }


class JacoTwoFingerGripper(JacoTwoFingerGripperBase):
    """
    Control both Jaco fingers with one incremental open/close action.
    """

    @property
    def init_action(self):
        """Hold both position actuators at their open target after reset."""
        return -np.ones(len(self.actuators))

    def format_action(self, action):
        """
        Move both actuator targets toward open (-1) or closed (+1).

        XML actuator ranges map the normalized targets to [0, 1.4] radians.

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + np.ones(2) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1
