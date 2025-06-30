""" Generate event data and video of the trajectory of a table tennis ball

This script contains the code to generate a hdf5 file of the event data as
well as a video of a table tennis ball that moves and rotates as specified
It can be used to generate the first iteration of datasets used for the 
spin-estimation model.

There are many parameters that can be adjusted to change the simulation
and the output video. The most important ones are:


"""
import bpy
import bpy_extras
from bpy_extras import view3d_utils
import os
import cv2
import tqdm as pbar
import numpy as np
import sys
from mathutils import Vector
import logging
sys.path.append("../src")
from dvs_sensor import *
from dvs_sensor_blender import Blender_DvsSensor
import rotations
import eventIO

# ---------------------------------------------------- Parameters ---------------------------------------------------

# paths:
path_scene = os.path.abspath(os.getcwd()) + "/../data/scenes/"
path_output = os.path.abspath(os.getcwd()) + "/../data/output/"
path_logs = os.path.abspath(os.getcwd()) + "/../data/output/logs/"
noise_paths = os.path.abspath(os.getcwd()) + "/../data/noise/"
noise_paths = (noise_paths + "noise_pos_161lux.npy", noise_paths + "noise_pos_161lux.npy") # two paths needed to initialize

# File names:
object_name = "bigger.blend"
ball_name = "Sphere"                   # Name of the object inside blender-scene
camera_name = "Camera"                 # ...
temp_name =   "temp/temp"                   # just the blender rendering file
log_name = "render.log"                # name of the log file
output_name = "spinning_ball"          # name for the hdf5 file and the video

# general settings:
generate_video = True
generate_hdf5 = True
generate_event_video = True            # Should a event video be generated as well
save_blender = False
simulate = True                        # set False if just the blender file is needed
random_rotation = True                 # set False if the ball should rotate as manually specified
random_rps = False                     # not supported yet
save_position = True                   # Should the position ad diameter in pixels be saved
finish_early = False                  # If True, the simulation will stop after 40 frames
fix_to_1_s = True                     # If True, the simulation will be fixed to 1 second, and the timestamps will be adjusted later

# Ground truth settings (save_position must be True)
render_box = False                      # Should we render the bounding box inside the video (Debug)
approx_size = 100                       # Approx size of the ball in pixels (TODO find better solution)

# Simulation settings:
total_frames = 500                     # high enough to cover rotation
rps = 80                               # rotations per second
total_rotations = 2                    # total rotations util the end of the simulation
video_length = total_rotations / rps   # length of the video in seconds
fps = int(total_frames / video_length) # frames per second
ball_speed = 0.5                       # how much the ball moves [m/s]
ball_start = (0, 0.4, 0)               # start position of the ball
ball_end = (0, -0.4, 0)                # end position of the ball
rotation_axis = (0, 1, 1)              # axis of rotation (only if not random rotation)
video_fps = 20                         # video will be slow-mo so it is actually viewable
simulation_samples = 64                # light rays per pixel that blender will use

# Camera settings: (position and rotation from blender scene)
resolution_x = 1280
resolution_y = 720
resolution_percentage = 100
focal_length = 9.0  # (mm)
pixel_pitch = 0.0075                     # Abstand zwischen pixeln im sensor (beinflusst FOV)

# Event Camera settings
th_pos = 0.1                            # on threshold 
th_neg = 0.1                            # off threshold
th_n = 0.13                             # noise threshold                                     
lat = 400                               # latency
tau = 400                               # time constant
jit = 100                               # jitter
bgn = 0.0001                            # background noise
ref_period = 50                         # ref-time of the event camera



# branch used: 3c4b99c

if fix_to_1_s:
    video_length = 1.0  # fix video length to 1 second
    fps = total_frames  # set fps to total frames


def init_scene():
    """ initialize blender scene

        returns:
            ball and scene
    """
    bpy.ops.wm.open_mainfile(filepath=path_scene + object_name)
    ball = bpy.data.objects[ball_name]
    scene = bpy.context.scene

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = total_frames
    bpy.context.scene.render.fps = fps
    bpy.context.scene.render.image_settings.file_format = 'PNG'


    print(f"Scene initialized with {total_frames} frames at {fps} fps and a video length of {video_length} seconds.")
    print(f"Ball has a spin with {total_rotations} rotations at {rps} rps.")

    # Make background white
    bpy.data.worlds["World"].use_nodes = True
    bg = bpy.data.worlds["World"].node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.1, 0.1, 0.1, 1)  # R, G, B, Alpha (black)
    bg.inputs[1].default_value = 0.0

    return ball


def init_camera():
    """ Initialize event and frame cameras

        returns event_camera
    """
    cam_pos = bpy.data.objects[camera_name].location
    cam_rot = bpy.data.objects[camera_name].rotation_euler
    event_camera = Blender_DvsSensor("Sensor")
    event_camera.cam = bpy.data.objects[camera_name]
    event_camera.set_sensor(nx=resolution_x, ny=resolution_y, pp=pixel_pitch)
    print(f"Initializing event camera with parameters: "
          f"th_pos={th_pos}, th_neg={th_neg}, th_n={th_n}, "
          f"lat={lat}, tau={tau}, jit={jit}, bgn={bgn}")
    event_camera.set_dvs_sensor(th_pos=th_pos, th_neg=th_neg, th_n=th_n, lat=lat, tau=tau, jit=jit, bgn=bgn)
    event_camera.ref = ref_period  # reduce refractory time to allow more frequent events
    print(f"Event camera initialized with refractory time of {event_camera.ref} us")
    event_camera.set_sensor_optics(focal_length)
    bpy.context.scene.render.resolution_x = event_camera.def_x
    bpy.context.scene.render.resolution_y = event_camera.def_y
    bpy.context.scene.render.resolution_percentage = resolution_percentage
    bpy.context.scene.eevee.taa_render_samples = simulation_samples

    bpy.context.scene.camera = event_camera.cam
    event_camera.set_position(cam_pos)
    event_camera.set_angle(cam_rot)
    event_camera.init_tension()
    event_camera.init_bgn_hist(noise_paths[0], noise_paths[1])
    return event_camera


def calculate_position_frame() -> int:
    """ Calculate the frame to set the end position to

        This is dependent on the speed of the ball and the fps
        This function calculates the frame when the position of the ball
        should be ball_end.
    """
    return total_frames



def create_rotations():
    """ Creates Rotations in a cubic way, as discussed with David

        TODO: move this function to the dataset-genration file, since it is relevant to the whole
        dataset and not for just one simulation

        A total amount of rotations is generated that have speeds in a specified range.
        set the parameters accordingly    
    """
    n = 28 # points per axis
    max_speed = 80   # maximum speed in rps
    min_speed = 5    # necessary?
    lin = np.linspace(-1, 1, n)
    x, y, z = np.meshgrid(lin, lin, lin)
    points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    # cut out speeds that are not needed
    distances = np.linalg.norm(points, axis=1)
    points = points[(distances <= 1) & (distances >= (min_speed / max_speed))]
    points = points * 80
    # set n to have enough samples 
    print(f"n: {n}, resulting rotations: {points.shape[0]}")
    # rotation can be created this way:
    r = rotations.Rotation()
    r.set_axis(points[0])
    return r



def generate_keyframes(ball, rotation: rotations.Rotation) -> None:
    """ Generate keyframes to simulate object
    
        For the rotation only the axis is needed because the simulation is fixed
        to "total_rotations" rotations
    """
    ball.rotation_mode = 'AXIS_ANGLE'
    ax = rotation.get_axis()
    ball.rotation_axis_angle = (0, ax[0], ax[1], ax[2])
    ball.keyframe_insert(data_path="rotation_axis_angle", frame=0, index=-1)

    ball.rotation_axis_angle = (total_rotations * np.pi * 2, ax[0], ax[1], ax[2])
    ball.keyframe_insert(data_path="rotation_axis_angle", frame=total_frames, index=-1)

    ball.location = ball_start
    ball.keyframe_insert(data_path="location", frame=0)

    ball.location = ball_end
    ball.keyframe_insert(data_path="location", frame=calculate_position_frame())

    # Set interpolation to linear for constant rotation speed
    for fcurve in ball.animation_data.action.fcurves:
        for kf in fcurve.keyframe_points:
            kf.interpolation = 'LINEAR'



def get_screen_positions(ball):
    ''' Returns the screen coords of the ball

        This Funciton should return the screen coords of the ball, to use it in the ground truth file
        The network should only get the ball-area as input

        At the moment this only calculates the position in pixels.
        Maybe it would be beneficial to also include the size of the ball ROI
        -> This was buggy in the last implementation so just let it be a parameter for now

    '''
    center = bpy_extras.object_utils.world_to_camera_view(
        scene=bpy.context.scene,
        obj=bpy.context.scene.camera,
        coord=ball.location
    )
    render = bpy.context.scene.render
    res_x = render.resolution_x * render.resolution_percentage / 100
    res_y = render.resolution_y * render.resolution_percentage / 100

    pixel_x = center.x * res_x
    pixel_y = (1 - center.y) * res_y

    return pixel_x, pixel_y



def simulate(event_camera, ball):
    """ Executes the simulation
    
    """

    print(f"Delta t: {1000000.0 * (1.0 / fps)} us")
    # redirect output to log file
    logfile = path_logs + log_name
    open(logfile, 'a').close()
    old = os.dup(sys.stdout.fileno())
    sys.stdout.flush()
    os.close(sys.stdout.fileno())
    fd = os.open(logfile, os.O_WRONLY)

    if simulate:
        if generate_video:
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
            video = cv2.VideoWriter(path_output + output_name + ".avi", fourcc, video_fps, (event_camera.def_x, event_camera.def_y))

        if generate_hdf5:
            # init Event Buffer
            ev = EventBuffer(0)
        
        for frame in pbar.tqdm(range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1), desc="Rendering frames"):
            bpy.context.scene.frame_set(frame)

            # file_name = path_output + temp_name + str(frame) + ".png"
            file_name = path_output + temp_name + ".png" # dont save that much images for now
            bpy.context.scene.render.filepath = file_name
            bpy.ops.render.render(write_still=True)
            img = cv2.imread(filename=file_name)

            # calculate ball-positon
            result = get_screen_positions(ball)
            if result and render_box:
                # temp -> draw bounding box
                x, y = result
                r = approx_size / 2
                top_left = (int(x - r), int(y - r))
                bot_right = (int(x + r), int(y + r))
                cv2.rectangle(img, top_left, bot_right, (0, 255, 0), 2)

            if generate_hdf5:
                if frame == 0:
                    event_camera.init_image(img)
                else:
                    delta_t = 1000000.0 * (1.0 / fps)  # delta t in us (1000000 us = 1 s)
                    pk = event_camera.update(img, delta_t)
                    ev.increase_ev(pk)

            if generate_video:
                video.write(img)

            if finish_early and frame >= 40:
                print("Simulation finished early after 40 frames.")
                break

        # disable output redirection
        os.close(fd)
        os.dup(old)
        os.close(old)

        if generate_video:
            video.release()

        if fix_to_1_s:
            # readjust the timestamps
            ev.sort()
            video_length = total_rotations / rps
            print(f"Adjusting timestamps to {video_length} seconds")
            print(f"Old timestamps: {ev.get_ts()[0]} - {ev.get_ts()[-1]}")
            print(f"Result: {int(ev.get_ts()[-1] * video_length)} and not rounded: {ev.get_ts()[-1] * video_length}")
            for i in range(len(ev.ts)):
                ev.ts[i] = int(ev.ts[i] * video_length)
            
            eventIO.print_event_info(ev)

        if generate_hdf5:
            # ev.write(path_output + output_name + ".dat")
            ev.sort()
            eventIO.save_hdf5(ev, path_output + output_name + ".hdf5")
            eventIO.print_event_info(ev)

        if generate_event_video:
            eventIO.create_video(ev, path_output + output_name + "_events.avi", (resolution_x, resolution_y), fps=video_fps, tw=50)

    print(f"Number events: {ev.i}")

    if save_blender:
        bpy.ops.wm.save_as_mainfile(filepath=path_output + output_name + ".blend")


if __name__ == "__main__":
    print(path_scene + object_name)
    ball = init_scene()
    event_camera = init_camera()
    rot = rotations.random_rotation()
    generate_keyframes(ball, rot)
    simulate(event_camera, ball)