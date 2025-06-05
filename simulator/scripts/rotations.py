"""  File to handle everything related to rotations

Includes a class to represent a rotation

This file contains code to generate a random rotation and use
it in every representation that could be useful. The generation works in a way that
every possible rotation is eaually probable.

This file conatins the following classes and functions:
    * Rotation: Class to represent a rotation
    *

Notes:
    * At the moment omega is completely useless, because the speed of the rotation is calculated 
      using the fps and a rounds per second value.
"""

import numpy as np

class Rotation:
    """Class to represent a rotation
    
    This class is used to represent a rotation in 3D space. It can be used to
    generate a random rotation or convert different representations of a rotation to each other.
    
    Attributes:
        phi (float): Phi part of the exponential coordinates representation [0 -> 2π].
        theta (float): Theta part of the exponential coordinates representation cos(theta) [-1 -> 1].
        omega (float): angle of the rotation [0 -> 2π].

    Methods:

    """

    def __init__(self):
        """Constructor for the Rotation class
        
        This constructor initializes the rotation matrix to the identity matrix.
        """
        self.phi = 0
        self.theta = 0
        self.omega = 0

    def __init__(self, x:float, y: float, z: float):
        """ Constructor for a Rotation
        
            With this Constructor, a Rotation is initialized with a given axis (x, y, z)
        """
        self.set_axis(x, y, z)

    def __init__(self, axis:np.ndarray):
        """ Create Rotation from axis
        
            Overload with numpy array instead of values
        """ 
        self.set_axis(axis[0], axis[1], axis[2])

    def get_axis(self) -> np.ndarray:
        """Get the axis of rotation
        
        This method returns the axis of rotation as a numpy array.
        
        Returns:
            np.ndarray: [x, y, z] Axis of rotation
        """
        return np.array([np.cos(self.phi) * np.cos(self.theta),
                         np.sin(self.phi) * np.cos(self.theta),
                         np.sin(self.theta)])
    
    def set_axis(self, x:float, y:float, z:float) -> None:
        """ Sets roation axis and speed
        
            The internal values are set in a way, that the rotation represents a
            rotation around the given axis with a speed of the maginitude of the vector
        """
        self.theta = np.arcsin(z)
        self.phi = np.arctan2(y, x)
        self.omega = np.linalg.norm([x,y,z])
    
    def get_angle(self) -> float:
        """Get the angle of rotation
        
        This method returns the angle of rotation in radians.
        
        Returns:
            float: Angle of rotation in radians
        """
        return self.omega
    
    

# TODO: Add other functions as needed

def random_rotation() -> Rotation:
    """Generate a random rotation
    
    This function generates a random rotation in 3D space. It uses the exponential coordinates
    representation of a rotation to generate a random rotation.
    
    Returns:
        Rotation: A random rotation object
    """
    rot = Rotation()
    rot.phi = np.random.uniform(0, 2 * np.pi)
    cos_theta = np.random.uniform(-1, 1)
    rot.theta = np.arccos(cos_theta)
    rot.omega = np.random.uniform(0, 2 * np.pi)
    return rot
