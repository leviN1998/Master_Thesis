import bpy
import sys
import numpy as np
import bpy_extras
import cv2
sys.path.append("../utils")
sys.path.append("../utils/IEBCS")
sys.path.append("../src/utils/IEBCS") # debug
sys.path.append("../src/utils")       # debug
from dvs_sensor import *
from dvs_sensor_blender import Blender_DvsSensor
import eventIO
import rotations
from rotations import Rotation
import yaml
import os
import time
import pandas as pd

class Simulator:
    def __init__(self, config, dataset_path, logger, simulation_nr=0, pid=0):
        self.logger = logger
        self.set_config(config)
        self.simulation_nr = simulation_nr
        self.dataset_path = dataset_path
        self.num_string = str(self.simulation_nr).zfill(5)
        self.output_name = self.dataset_path + f"data/{self.num_string}/{self.num_string}_"
        self.ball_coords = []
        self.pid = pid

        try:
            os.mkdir(self.dataset_path + "data/" + self.num_string)
        except FileExistsError:
            self.logger.error(f"Directory {self.dataset_path}data/{self.num_string} already exists. Please remove it before running the simulation again.")

    def set_config(self, config):
        self.generate_video = config["generate_video"]
        self.genrate_hdf5 = config["generate_hdf5"]
        self.generate_event_video = config["generate_event_video"]
        self.save_blender = config["save_blender"]
        
        self.ramdom_orientation = config["random_orientation"]
        self.initial_orientation = config["initial_orientation"]
        self.spin_axis = config["spin_axis"]

        self.ball_start = config["ball_start"]
        self.ball_end = config["ball_end"]
        
        self.frames = config["frames"]
        self.simulation_time = config["simulation_time"]
        self.video_fps = config["video_fps"]
        self.simulation_samples = config["simulation_samples"]
        self.ball_name = config["ball_name"]

        self.resolution_x = config["resolution_x"]
        self.resolution_y = config["resolution_y"]
        self.resolution_percentage = config["resolution_percentage"]
        self.focal_length = config["focal_length"]
        self.pixel_pitch = config["pixel_pitch"]

        self.th_pos = config["th_phos"]
        self.th_neg = config["th_neg"]
        self.th_n = config["th_n"]
        self.lat = config["lat"]
        self.tau = config["tau"]
        self.jit = config["jit"]
        self.bgn = config["bgn"]
        self.ref_period = config["ref_period"]

    
    def run_simmulation(self):
        self.init_scene()
        self.init_camera()
        self.generate_keyframes()
        self.simulate()
        self.logger.info(f"Simulation {self.simulation_nr} finished. Output saved to {self.output_name}")
    

    def init_scene(self):
        