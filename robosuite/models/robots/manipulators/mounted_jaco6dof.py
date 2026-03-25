import numpy as np
from robosuite.models.robots.manipulators.manipulator_model import (
    ManipulatorModel,
)
from robosuite.utils.mjcf_utils import xml_path_completion


class MountedJaco6DOF(ManipulatorModel):
    """
    Libero-spatial assumes mounted robot. They prepend ``Mounted`` to the 
    actual robot name (see Libero_Tabletop_Manipulation). When creating an 
    environment, pass in robots=["Jaco6DOF"] and libero will invoke this class.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        # TODO: Make the robot body black to look like our real-world Jaco
        super().__init__(
            xml_path_completion("robots/jaco6dof/robot.xml"), idn=idn
        )

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        # TODO: Define our own gripper. Do not use JacoThreeFingerGripper as 
        # reference since that gripper has 6 DOF, compared to PandaGripper's 2.
        # We don't want too many DOFs on gripper or else we'd have to modify 
        # VLA's output space.
        return "PandaGripper"

    @property
    def default_controller_config(self):
        return "default_jaco"

    @property
    def init_qpos(self):
        return np.array([-1.5, 2.93, 1, -2.09, 1.44, 1.32])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
