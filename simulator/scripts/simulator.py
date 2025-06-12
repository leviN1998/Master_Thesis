""" Simulate one sample

    This class represents the simulator and is able to simulate one sample at a time.
    It is used by the simulate.py script to manage and divide the simulation process.
    The class is responsible for loading the scene, setting up the camera, and running the simulation.

"""

import bpy
import sys
sys.path.append("../src")
from dvs_sensor import *
from dvs_sensor_blender import Blender_DvsSensor


class Simulator:
    def __init__(self, config, simulation_nr=0):
        self.set_config(config)
        self.simulation_nr = simulation_nr

    
    def set_config(self, config):
        """ Set the configuration for the simulation """
        
        self.generate_video = config["generate_video"]
        self.generate_hdf5 = config["generate_hdf5"]
        self.generate_event_video = config["generate_event_video"]
        self.save_blender = config["save_blender"]
        self.random_orientation = config["random_orientation"]

        self.total_frames = config["total_frames"]
        self.total_rotations = config["total_rotations"]
        # Video length can be calculated from rps: len = total_rotations / rps
        # fps can be calculated with video length: fps = int(total_frames / video_length)
        self.ball_speed = config["ball_speed"]
        self.ball_start = config["ball_start"]
        self.ball_end = config["ball_end"]
        self.video_fps = config["video_fps"]
        self.simulation_samples = config["simulation_samples"]

        self.resolution_x = config["resolution_x"]
        self.resolution_y = config["resolution_y"]
        self.resolution_percentage = config["resolution_percentage"]
        self.focal_length = config["focal_length"]
        self.pixel_pitch = config["pixel_pitch"]


    def init_scene(self):
        """ Initialize the scene for the simulation """
        pass
        
    def init_camera(self):
        """ Set up ev camera """
        pass

    def calculate_position_frame(self) -> int:
        """ Calculate the frame to set the end position to

            This is dependent on the speed of the ball and the fps
            This function calculates the frame when the position of the ball
            should be ball_end.
        """
        return 0
    
    def genrate_keyframes(self):
        """ Generate keyframes to simulate object
    
        For the rotation only the axis is needed because the simulation is fixed
        to "total_rotations" rotations
        
        """
        pass


    def get_screen_positions(self):
        ''' Returns the screen coords of the ball

            This Funciton should return the screen coords of the ball, to use it in the ground truth file
            The network should only get the ball-area as input

            At the moment this only calculates the position in pixels.
            Maybe it would be beneficial to also include the size of the ball ROI
            -> This was buggy in the last implementation so just let it be a parameter for now

        '''
        pass


    def simulate(self):
        """ Run the simulation

            This function runs the simulation and saves the results to the specified files.
            It is responsible for setting up the scene, camera, and running the simulation.
        """
        pass