""" Simulate one sample

    This class represents the simulator and is able to simulate one sample at a time.
    It is used by the simulate.py script to manage and divide the simulation process.
    The class is responsible for loading the scene, setting up the camera, and running the simulation.

"""

import bpy
import sys
import numpy as np
import bpy_extras
import cv2
sys.path.append("../src")
from dvs_sensor import *
from dvs_sensor_blender import Blender_DvsSensor
import eventIO
import rotations
from rotations import Rotation
import yaml
import os
import time
import pandas as pd


def get_num_string(number: int):
    """ Returns a string representation of the number with leading zeros (we now take 5 digits ~ 10 000)

        Args:
            number (int): The number to convert

        Returns:
            str: The string representation of the number with leading zeros
    """
    return str(number).zfill(5)


class Simulator:
    def __init__(self, config, dataset_path, spin:Rotation, initial_rot: Rotation, logger, simulation_nr=0):
        self.logger = logger
        self.set_config(config)
        self.simulation_nr = simulation_nr
        self.dataset_path = dataset_path
        self.set_spin(spin)
        self.initial_rot = initial_rot
        self.num_string = get_num_string(self.simulation_nr)
        self.output_name = self.dataset_path + f"data/{self.num_string}/{self.num_string}_"
        self.ball_coords = []

    
    def set_config(self, config):
        """ Set the configuration for the simulation """
        
        self.generate_video = config["generate_video"]
        self.generate_hdf5 = config["generate_hdf5"]
        self.generate_event_video = config["generate_event_video"]
        self.save_blender = config["save_blender"]
        self.random_orientation = config["random_orientation"]
        self.fix_to_1_s = config["fix_to_1_s"]

        self.total_frames = config["total_frames"]
        self.total_rotations = config["total_rotations"]
        # Video length can be calculated from rps: len = total_rotations / rps
        # fps can be calculated with video length: fps = int(total_frames / video_length)
        self.ball_speed = config["ball_speed"]
        self.ball_start = config["ball_start"]
        self.ball_end = config["ball_end"]
        self.video_fps = config["video_fps"]
        self.simulation_samples = config["simulation_samples"]
        self.ball_name = config["ball_name"]

        self.resolution_x = config["resolution_x"]
        self.resolution_y = config["resolution_y"]
        self.resolution_percentage = config["resolution_percentage"]
        self.focal_length = config["focal_length"]
        self.pixel_pitch = config["pixel_pitch"]

        self.th_pos = config["th_pos"]
        self.th_neg = config["th_neg"]
        self.th_n = config["th_n"]
        self.lat = config["lat"]
        self.tau = config["tau"]
        self.jit = config["jit"]
        self.bgn = config["bgn"]
        self.ref_period = config["ref_period"]


    def set_spin(self, spin:Rotation):
        """ Set the spin of the ball for the simulation 
        
            If changed, some values need to be recalculated, such as the
            fps, video length and the frame to set the end position to.

            init_scene() and init_camera() need to be called again
            to apply the changes to the scene and camera.
        """

        self.spin = spin
        self.video_length = self.total_rotations / self.spin.get_angle()
        self.fps = int(self.total_frames / self.video_length)
        if self.fix_to_1_s:
            self.video_length = 1.0
            self.fps = self.total_frames
            self.logger.debug(f"Fixing video length to 1s, setting fps to total_frames real length is: {self.total_rotations / self.spin.get_angle()}")
        self.logger.debug(f"Set spin: {self.spin.get_axis()} with angle: {self.spin.get_angle()} and omega: {self.spin.get_angle()}")
        self.logger.debug(f"Calculated video length: {self.video_length} seconds with fps: {self.fps} and total frames: {self.total_frames}")


    def run_simulation(self):
        """ Run the simulation

            This function initializes the scene, camera and generates the keyframes.
            It then runs the simulation and saves the results to the specified output path.
        """
        self.init_scene()
        self.init_camera()
        self.generate_keyframes()
        self.simulate()
        self.logger.info(f"Simulation {self.simulation_nr} finished. Output saved to {self.output_name}")


    def init_scene(self):
        """ Initialize the scene for the simulation """

        bpy.ops.wm.open_mainfile(filepath=self.dataset_path + "config/scene.blend")
        self.ball = bpy.data.objects[self.ball_name]
        self.scene = bpy.context.scene

        self.scene.frame_start = 0
        self.scene.frame_end = self.total_frames
        self.scene.render.fps = self.fps
        self.scene.render.image_settings.file_format = 'PNG'

        # set background
        bpy.data.worlds["World"].use_nodes = True
        bg = bpy.data.worlds["World"].node_tree.nodes["Background"]
        bg.inputs[0].default_value = (0.1, 0.1, 0.1, 1)  # R, G, B, Alpha (black)
        bg.inputs[1].default_value = 0.0
        

        
    def init_camera(self):
        """ Set up ev camera """
        cam_pos = bpy.data.objects["Camera"].location
        cam_rot = bpy.data.objects["Camera"].rotation_euler
        self.event_camera = Blender_DvsSensor("Sensor")
        self.event_camera.cam = bpy.data.objects["Camera"]
        self.event_camera.set_sensor(nx=self.resolution_x, ny=self.resolution_y, pp=self.pixel_pitch)
        self.event_camera.set_dvs_sensor(th_pos=self.th_pos, th_neg=self.th_neg, th_n=self.th_n, lat=self.lat, tau=self.tau, jit=self.jit, bgn=self.bgn)
        self.event_camera.ref = self.ref_period
        self.event_camera.set_sensor_optics(self.focal_length)
        self.scene.render.resolution_x = self.resolution_x
        self.scene.render.resolution_y = self.resolution_y
        self.scene.render.resolution_percentage = self.resolution_percentage
        self.scene.eevee.taa_render_samples = self.simulation_samples

        self.scene.camera = self.event_camera.cam
        self.event_camera.set_position(cam_pos)
        self.event_camera.set_angle(cam_rot)
        self.event_camera.init_tension()
        self.event_camera.init_bgn_hist(self.dataset_path + "noise/noise_pos_161lux.npy", self.dataset_path + "noise/noise_pos_161lux.npy")


    def calculate_position_frame(self) -> int:
        """ Calculate the frame to set the end position to

            This is dependent on the speed of the ball and the fps
            This function calculates the frame when the position of the ball
            should be ball_end.

            Falsch rum!! Der end-frame muss irgendwie max-frames sein...
            nochmal anschauen, weil auch nur den halben Bildschrim zu durchqueren nicht so viel sinn macht.
        """
        distance = np.linalg.norm(np.array(self.ball_end) - np.array(self.ball_start))
        max_speed = distance / self.video_length
        self.logger.debug(f"Max speed: {max_speed} for distance: {distance} and video length: {self.video_length}")
        if self.ball_speed > max_speed:
            self.logger.error(f"Ball speed is too high for the given video length. Please adjust the parameters. [{self.ball_speed} > {max_speed}]")
            # sys.exit(1)

        end_frame = int(self.total_frames * (self.ball_speed / max_speed))
        self.logger.debug(f"Calculated end frame: {end_frame} for ball speed: {self.ball_speed} and max speed: {max_speed}")
        return end_frame
    
    
    def generate_keyframes(self):
        """ Generate keyframes to simulate object
    
        For the rotation only the axis is needed because the simulation is fixed
        to "total_rotations" rotations
        
        """
        self.ball.rotation_mode = 'AXIS_ANGLE'
        if self.random_orientation:
            self.initial_rot = rotations.random_rotation()

        ax = self.initial_rot.get_axis()
        angle = self.initial_rot.get_angle() # this is in degrees
        self.logger.debug(f"Initial rotation axis: {ax} with {angle} deg.")
        angle = angle * np.pi / 180.0 # convert to radians
        self.ball.rotation_axis_angle = (angle, ax[0], ax[1], ax[2])
        # apply initial rotation
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

        ax = self.spin.get_axis()
        self.ball.rotation_axis_angle = (0, ax[0], ax[1], ax[2])
        self.ball.keyframe_insert(data_path="rotation_axis_angle", frame=0, index=-1)

        self.ball.rotation_axis_angle = (self.total_rotations * np.pi * 2, ax[0], ax[1], ax[2])
        self.ball.keyframe_insert(data_path="rotation_axis_angle", frame=self.total_frames, index=-1)

        # TODO: fix for realistic speeds
        self.ball.location = self.ball_start
        self.ball.keyframe_insert(data_path="location", frame=0)

        self.ball.location = self.ball_end
        self.ball.keyframe_insert(data_path="location", frame=self.total_frames)
        self.calculate_position_frame() # just to log the value

        # Set interpolation to linear for constant rotation speed
        for fcurve in self.ball.animation_data.action.fcurves:
            for kf in fcurve.keyframe_points:
                kf.interpolation = 'LINEAR'



    def get_screen_positions(self):
        ''' Returns the screen coords of the ball

            This Funciton should return the screen coords of the ball, to use it in the ground truth file
            The network should only get the ball-area as input

            At the moment this only calculates the position in pixels.
            Maybe it would be beneficial to also include the size of the ball ROI
            -> This was buggy in the last implementation so just let it be a parameter for now

        '''
        center = bpy_extras.object_utils.world_to_camera_view(
            scene=self.scene,
            obj=self.scene.camera,
            coord=self.ball.location
        )
        render = self.scene.render
        res_x = render.resolution_x * render.resolution_percentage / 100
        res_y = render.resolution_y * render.resolution_percentage / 100

        pixel_x = center.x * res_x
        pixel_y = (1 - center.y) * res_y

        return pixel_x, pixel_y
    

    def update_ground_truth(self, frame):
        """ Update the ground truth file with the current frame's data

            This function updates the pixel coords of the ball at the given frame.
        """
        
        self.ball_coords.append((frame, self.get_screen_positions()))


    def save_ground_truth(self):
        """ Save the ground truth data to a file
        """
        ground_truth_path = self.output_name + "ground_truth.yaml"
        self.logger.info(f"Saving ground truth data to {ground_truth_path}")
        # Save ground truth (rotation info) as CSV
        gt = {
            "rotation_x": [self.spin.get_axis()[0]],
            "rotation_y": [self.spin.get_axis()[1]],
            "rotation_z": [self.spin.get_axis()[2]],
            "rotation_omega": [self.spin.get_angle()],
        }
        gt_df = pd.DataFrame(gt)
        gt_path = self.output_name + "ground_truth.csv"
        gt_df.to_csv(gt_path, index=False)
        self.logger.info(f"Ground truth data saved to {gt_path}")

        # Save metadata as CSV
        metadata = {
            "rotation_x": [self.spin.get_axis()[0]],
            "rotation_y": [self.spin.get_axis()[1]],
            "rotation_z": [self.spin.get_axis()[2]],
            "rotation_omega": [self.spin.get_angle()],
            "ball_start_x": [self.ball_start [0]],
            "ball_start_y": [self.ball_start[1]],
            "ball_start_z": [self.ball_start[2]],
            "ball_end_x": [self.ball_end[0]],
            "ball_end_y": [self.ball_end[1]],
            "ball_end_z": [self.ball_end[2]],
            "ball_speed": [self.ball_speed],
            "initial_rot_x": [self.initial_rot.get_axis()[0]],
            "initial_rot_y": [self.initial_rot.get_axis()[1]],
            "initial_rot_z": [self.initial_rot.get_axis()[2]],
            "ball_position_world_x": [self.ball.location[0]],
            "ball_position_world_y": [self.ball.location[1]],
            "ball_position_world_z": [self.ball.location[2]],
            "camera_position_world_x": [self.event_camera.cam.location[0]],
            "camera_position_world_y": [self.event_camera.cam.location[1]],
            "camera_position_world_z": [self.event_camera.cam.location[2]],
            "camera_rotation_world_x": [self.event_camera.cam.rotation_euler[0]],
            "camera_rotation_world_y": [self.event_camera.cam.rotation_euler[1]],
            "camera_rotation_world_z": [self.event_camera.cam.rotation_euler[2]],
        }
        metadata_df = pd.DataFrame(metadata)
        metadata_path = self.output_name + "metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        self.logger.info(f"Metadata saved to {metadata_path}")

        # Save ball coordinates per frame as CSV
        coords_df = pd.DataFrame(self.ball_coords, columns=["frame", "screen_position"])
        # Split screen_position tuple into two columns
        coords_df[["screen_x", "screen_y"]] = pd.DataFrame(coords_df["screen_position"].tolist(), index=coords_df.index)
        coords_df = coords_df.drop(columns=["screen_position"])
        coords_path = self.output_name + "ball_coords.csv"
        coords_df.to_csv(coords_path, index=False)
        self.logger.info(f"Ball coordinates saved to {coords_path}")

    
    def redircet_output(self):
        """ Rediret the blender output to a file

            This function redirects the blender output to a file, so that the logs can be saved
            and used for debugging purposes.
        """
        logfile = self.dataset_path + "tmp/render.log"
        with open(logfile, 'a') as f:
            f.write("\n\n========== NEW BLENDER OUTPUT ==========\n\n")
            f.write(f"Simulation {self.simulation_nr} started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n\n")
        self.old = os.dup(sys.stdout.fileno())
        sys.stdout.flush()
        os.close(sys.stdout.fileno())
        self.fd = os.open(logfile, os.O_WRONLY | os.O_APPEND)


    def restore_output(self):
        os.close(self.fd)
        os.dup(self.old)
        os.close(self.old)


    def simulate(self):
        """ Run the simulation

            This function runs the simulation and saves the results to the specified files.
            It is responsible for setting up the scene, camera, and running the simulation.
        """
        self.logger.info(f"Starting simulation {self.simulation_nr} with spin: {self.spin.get_angle()} rps")
        self.logger.debug(f"Delta t calculated as: {1000000.0 * (1.0 / self.fps)} us with fps: {self.fps} ")

        if self.generate_video:
            self.logger.debug("Generating video is enabled!")
            fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
            video = cv2.VideoWriter(self.output_name + "frames.avi", fourcc, self.video_fps, (self.resolution_x, self.resolution_y))

        if self.generate_hdf5:
            self.logger.debug("Generating hdf5 is enabled!")
            ev = EventBuffer(0)

        self.redircet_output()
        start_ts = time.time()
        end_ts = time.time()
        for frame in range(self.scene.frame_start, self.scene.frame_end + 1):
            duration = end_ts - start_ts
            start_ts = time.time()

            self.logger.progress(f"Simulation {self.simulation_nr}: Rendering frame {frame}/{self.scene.frame_end}  ({duration:.2f} s/frame)")
            self.scene.frame_set(frame)

            file_name = self.dataset_path + "tmp/image_tmp.png"
            self.scene.render.filepath = file_name
            bpy.ops.render.render(write_still=True)
            img = cv2.imread(file_name)

            self.update_ground_truth(frame)
            if self.generate_video:
                video.write(img)

            if self.generate_hdf5:
                if frame == 0:
                    self.event_camera.init_image(img)
                else:
                    delta_t = 1000000.0 * (1.0 / self.fps)  # delta t in us (1000000 us = 1 s)
                    pk = self.event_camera.update(img, delta_t)
                    ev.increase_ev(pk)
            end_ts = time.time()

        self.restore_output()
        try:
            os.mkdir(self.dataset_path + "data/" + self.num_string)
        except FileExistsError:
            self.logger.error(f"Directory {self.dataset_path}data/{self.num_string} already exists. Please remove it before running the simulation again.")

        if self.generate_video:
            video.release()
            self.logger.info(f"Video saved to {self.output_name}frames.avi")

        if self.generate_hdf5:
            if self.fix_to_1_s:
                self.video_length = self.total_rotations / self.spin.get_angle()
                self.logger.info(f"Recalculating timestamps to fit video length of {self.video_length} seconds")
                self.logger.info(f"Result length is: {ev.get_ts()[-1] * self.video_length - ev.get_ts()[0] * self.video_length} us")
                for i in range(ev.i):
                    ev.ts[i] = int(ev.ts[i] * self.video_length)

            eventIO.save_hdf5(ev, self.output_name + "events.hdf5")
            self.logger.info(f"Events saved to {self.output_name}events.hdf5")

        if self.generate_event_video:
            eventIO.create_video(ev, self.output_name + "events.avi", (self.resolution_x, self.resolution_y), self.video_fps, tw=200)
            self.logger.info(f"Event video saved to {self.output_name}events.avi")

        if self.save_blender:
            bpy.ops.wm.save_as_mainfile(filepath=self.output_name + "scene.blend")
            self.logger.info(f"Blender scene saved to {self.output_name}scene.blend")

        self.save_ground_truth()