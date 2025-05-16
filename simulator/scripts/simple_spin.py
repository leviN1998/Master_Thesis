"""
    This is a first test to generate a simulation using blender

    The cube is loaded from a -blend scene and rotates as specified in the script
    The rendering takes place in a loop, where it would be possible to insert the event simulation

    A video is generated from the rendered images
"""

import bpy
import os
import cv2
import tqdm
import numpy as np
import contextlib
import sys


# set up paths
path_scene = os.path.abspath(os.getcwd()) + "/../data/scenes/" + "basic_cube.blend"
path_output = os.path.abspath(os.getcwd()) + "/../data/output/"
path_image = path_output + "render_img.png"
path_log = os.path.abspath(os.getcwd()) + "/../data/logs/"

# load scene
bpy.ops.wm.open_mainfile(filepath=path_scene)

# get cube
cube = bpy.data.objects['Cube']

# transform cube
cube.location = (0, 0, 0)
cube.scale = (0.5, 0.5, 0.5)

# Set up light
# bpy.ops.object.light_add(type='SUN', location=(10, 10, 10))

# animation settings
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 30
scene.render.fps = 15


cube.rotation_euler = (0, 0, 0)
cube.keyframe_insert(data_path="rotation_euler", frame=0)

cube.rotation_euler = (0, 0, 6.28319)  # 2 * pi = 360 degrees in radians
cube.keyframe_insert(data_path="rotation_euler", frame=120)

# Set interpolation to linear for constant rotation speed
for fcurve in cube.animation_data.action.fcurves:
    for kf in fcurve.keyframe_points:
        kf.interpolation = 'LINEAR'


# Initialize video writer
fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
video = cv2.VideoWriter(path_output + "spinning_cube.avi", fourcc, 30.0, (1920, 1080))

# Render settings
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = path_image
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.resolution_percentage = 100  # Ensure full resolution

# redirect output to log file
logfile = path_log + 'blender_render.log'
open(logfile, 'a').close()
old = os.dup(sys.stdout.fileno())
sys.stdout.flush()
os.close(sys.stdout.fileno())
fd = os.open(logfile, os.O_WRONLY)


# Render loop for each frame
for frame in tqdm.tqdm(range(scene.frame_start, scene.frame_end + 1), desc="Rendering frames"):
    scene.frame_set(frame)

    
    bpy.ops.render.render(write_still=True)

    # Read the rendered image
    img = cv2.imread(path_image)

    # Write the image to the video file
    video.write(img)

# Release the video writer
video.release()

# disable output redirection
os.close(fd)
os.dup(old)
os.close(old)