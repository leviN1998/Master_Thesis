import bpy
import cv2
import time
import os
import numpy as np
import eventIO
from dvs_sensor import *
from dvs_sensor_blender import Blender_DvsSensor


class Simulator:

    def __init__(self, config, logger, simulation_nr=0, pid=0):
        self.logger = logger
        self.set_config(config)
        self.simulation_nr = simulation_nr
        self.pid = pid
        self.num_string = str(simulation_nr).zfill(5)
        self.tmp_path = self.dataset_path + f"tmp/pid_{self.pid}/image_tmp.png"
        self.output_name = self.dataset_path + f"data/{self.num_string}/{self.num_string}_"
        self.scene_path = self.dataset_path + f"config/scene.blend"
        self.ball_coords = []

        try:
            os.mkdir(self.dataset_path + "data/" + self.num_string)
        except FileExistsError:
            self.logger.error(f"Directory {self.dataset_path}data/{self.num_string} already exists. Please remove it before running the simulation again.")


    def set_config(self, config):
        
        
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


    def init_scene(self):
        """
        
        """
        bpy.ops.wm.open_mainfile(filepath=self.scene_path)
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
        # print(self.th_pos, self.th_neg, self.th_n, self.lat, self.tau, self.jit, self.bgn, self.ref_period)
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


    def generate_spin_keyframes(self):
        self.ball.rotation_mode = 'AXIS_ANGLE'
        ax = self.initial_rot.get_axis()
        angle = self.initial_rot.get_angle()
        self.logger.debug(f"Initial rotation axis: {ax} with {angle} deg.")
        angle = angle * np.pi / 180.0 # convert to radians
        self.ball.rotation_axis_angle = (angle, ax[0], ax[1], ax[2])
        # apply initial rotation
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)




    def generate_position_keyframes(self):
        pass


    def get_screen_positions(self):
        pass


    def update_ground_truth(self, frame):
        pass


    def save_ground_truth(self):
        pass


    def redirect_output(self):
        pass

    
    def restore_output(self):
        pass


    def simulate(self):
        """
        
        """

        if self.generate_video:
            fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
            video = cv2.VideoWriter(self.output_name + "frames.avi", fourcc, self.video_fps, (self.resolution_x, self.resolution_y))

        ev = EventBuffer(0)

        self.redirect_output()
        start_ts = time.time()
        end_ts = time.time()
        for frame in range(self.scene.frame_start, self.scene.frame_end+1):
            duration = end_ts - start_ts
            start_ts = time.time()

            if frame % 100 == 0:
                self.logger.progress(f"Simulation {self.simulation_nr}: Rendering frame {frame}/{self.scene.frame_end}  ({duration:.2f} s/frame, {int(duration*self.total_frames)}s total.)")

            self.scene.render.filepath = self.tmp_path
            bpy.ops.render.render(write_still=True)
            img = cv2.imread(self.tmp_path)

            self.update_ground_truth(frame)
            if self.generate_video:
                video.write(img)

            if frame == 0:
                self.event_camera.init_image(img)
            else:
                delta_t = 1000000.0 * (1.0 / self.fps)  # delta t in us (1000000 us = 1 s)
                pk = self.event_camera.update(img, delta_t)
                ev.increase_ev(pk)
            end_ts = time.time()

        self.restore_output()

        if self.generate_video:
            video.release()
            self.logger.debug(f"Video saved to {self.output_name}frames.avi")

        bias = [self.th_pos, self.th_neg, self.th_n, self.lat, self.tau, self.jit, self.bgn, self.ref_period]
        eventIO.save_hdf5(ev, self.output_name + "events.hdf5", bias)

        self.save_ground_truth()


    def run_simulation(self):
        """
        Run the simulation.
        """
        self.init_scene()
        self.init_camera()
        self.generate_spin_keyframes()
        self.generate_position_keyframes()
        self.simulate()
        self.logger.info(f"Simulation {self.simulation_nr} finished. Output saved to {self.output_name}")


if __name__ == "__main__":
    print("test simulator module")