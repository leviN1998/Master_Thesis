"""
    This is a first test to generate a simulation using blender

    The ball is loaded from a -blend scene and rotates as specified in the script
    The rendering takes place in a loop, where it would be possible to insert the event simulation

    I first try of setting up the event camera is also implemented

    A video is generated from the rendered images
"""

import bpy
import os
import cv2
import tqdm as pbar
import numpy as np
import sys
from mathutils import Vector
sys.path.append("../src")
from dvs_sensor import *
from dvs_sensor_blender import Blender_DvsSensor
from event_display import EventDisplay


# set up paths
object_name = "basic_cube.blend"
path_scene = os.path.abspath(os.getcwd()) + "/../data/scenes/" + object_name
path_output = os.path.abspath(os.getcwd()) + "/../data/output/"
path_image = path_output + "render_img.png"
path_log = os.path.abspath(os.getcwd()) + "/../data/logs/"

# load scene
bpy.ops.wm.open_mainfile(filepath=path_scene)

# get cube
ball = bpy.data.objects['Sphere']

# transform ball
ball.location = (0, 0, 0)
ball.scale = (1, 1, 1)

# Set up light
if "Light" in bpy.data.objects:
    light = bpy.data.objects['Light']
    light.location = Vector((5, 5, 5))
    light.data.energy = 5

    light.data.type = 'SUN'

# animation settings
scene = bpy.context.scene
scene.frame_start = 0
scene.frame_end = 200
scene.render.fps = 10

# test with axis-angle rotation
ball.rotation_mode = 'AXIS_ANGLE'
ball.rotation_axis_angle = (0, 0, 0, -1)
ball.keyframe_insert(data_path="rotation_axis_angle", frame=0, index=-1)

ball.rotation_axis_angle = (np.pi*3, 0, 0, -1)
ball.keyframe_insert(data_path="rotation_axis_angle", frame=200, index=-1)

# Set interpolation to linear for constant rotation speed
for fcurve in ball.animation_data.action.fcurves:
    for kf in fcurve.keyframe_points:
        kf.interpolation = 'LINEAR'


# Render settings
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = path_image
render_res_x = 1920
render_res_y = 1080
scene.render.resolution_percentage = 100  # Ensure full resolution

# delete default camera
if "Camera" in bpy.data.objects:
    cam_pos = bpy.data.objects['Camera'].location
    cam_rot = bpy.data.objects['Camera'].rotation_euler
    bpy.data.objects['Camera'].select_set(True)
    bpy.ops.object.delete()

# set up event camera
ppsee = Blender_DvsSensor("Sensor")
ppsee.set_sensor(nx=render_res_x, ny=render_res_y, pp=0.015)
ppsee.set_dvs_sensor(th_pos=0.15, th_neg=0.15, th_n=0.05, lat=500, tau=300, jit=100, bgn=0.0001)
ppsee.set_sensor_optics(1)
master_collection = bpy.context.collection


master_collection.objects.link(ppsee.cam)
scene.camera = ppsee.cam

ppsee.set_position(cam_pos)
ppsee.set_angle(cam_rot)
ppsee.set_speeds([0.0, 0, 0], [0.0, 0.0, 10])

ppsee.init_tension()
ppsee.init_bgn_hist("../data/noise/noise_pos_161lux.npy", "../data/noise/noise_pos_161lux.npy")


scene.render.resolution_x = ppsee.def_x
scene.render.resolution_y = ppsee.def_y


# redirect output to log file
logfile = path_log + 'blender_render.log'
open(logfile, 'a').close()
old = os.dup(sys.stdout.fileno())
sys.stdout.flush()
os.close(sys.stdout.fileno())
fd = os.open(logfile, os.O_WRONLY)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
video = cv2.VideoWriter(path_output + "spinning_cube.avi", fourcc, 20.0, (ppsee.def_x, ppsee.def_y))

ev = EventBuffer(0)
ed = EventDisplay("Events", ppsee.def_x, ppsee.def_y, 10000)

# Render loop for each frame
for frame in pbar.tqdm(range(scene.frame_start, scene.frame_end + 1), desc="Rendering frames"):
    scene.frame_set(frame)

    ppsee.update_time(1 / 1000)
    ppsee.print_position()
    
    bpy.ops.render.render(write_still=True)

    # Read the rendered image
    img = cv2.imread(path_image)

    if frame == 0:
        ppsee.init_image(img)
    else:
        pk = ppsee.update(img, 1000)
        ed.update(pk, 1000)
        ev.increase_ev(pk)
        bpy.data.objects['Light'].data.energy += 0.01

    # Write the image to the video file
    video.write(img)

# Release the video writer
video.release()
ev.write(path_output + "spinning_cube_events.dat")

# disable output redirection
os.close(fd)
os.dup(old)
os.close(old)