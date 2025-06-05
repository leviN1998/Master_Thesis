"""
This file can be used to debug the configuratoin wihtout event cameras
Easier usage of blender



"""


import bpy
import bpy_extras
import os
import numpy as np
import rotations
import cv2
import tqdm as pbar
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view


path_scene = os.path.abspath(os.getcwd()) + "/../data/scenes/ball_moving.blend"
path_output = os.path.abspath(os.getcwd()) + "/../data/output/"

resolution_x = 1280
resolution_y = 720

frames = 100

ball_start = (-3, -3, 0)               # start position of the ball
ball_end = (3, 5, 0)                   # end position of the ball
# ball_start = (0, 2.7, 0)               # start position of the ball
# ball_end = (0, -2.7, 0)                   # end position of the ball



bpy.ops.wm.open_mainfile(filepath=path_scene)
ball = bpy.data.objects["Sphere"]
scene = bpy.context.scene

scene.frame_start = 0
scene.frame_end = frames
scene.render.fps = 30
scene.render.image_settings.file_format = 'PNG'

bpy.context.scene.render.resolution_x = resolution_x
bpy.context.scene.render.resolution_y = resolution_y
bpy.context.scene.render.resolution_percentage = 100

ball.rotation_mode = 'AXIS_ANGLE'
rotation = rotations.random_rotation()
ax = rotation.get_axis()
ball.rotation_axis_angle = (0, ax[0], ax[1], ax[2])
ball.keyframe_insert(data_path="rotation_axis_angle", frame=0, index=-1)

ball.rotation_axis_angle = (2 * np.pi * 2, ax[0], ax[1], ax[2])
ball.keyframe_insert(data_path="rotation_axis_angle", frame=frames, index=-1)

ball.location = ball_start
ball.keyframe_insert(data_path="location", frame=0)

ball.location = ball_end
ball.keyframe_insert(data_path="location", frame=frames)

# Set interpolation to linear for constant rotation speed
for fcurve in ball.animation_data.action.fcurves:
    for kf in fcurve.keyframe_points:
        kf.interpolation = 'LINEAR'


def get_screen_positions(ball):
    center = bpy_extras.object_utils.world_to_camera_view(
        scene=bpy.context.scene,
        obj=bpy.context.scene.camera,
        coord=ball.location
    )
    res_x = scene.render.resolution_x * scene.render.resolution_percentage / 100
    res_y = scene.render.resolution_y * scene.render.resolution_percentage / 100

    pixel_x = center.x * res_x
    pixel_y = (1 - center.y) * res_y

    return pixel_x, pixel_y


# Initialize video writer
fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
video = cv2.VideoWriter(path_output + "video" + ".avi", fourcc, 30, (resolution_x, resolution_y))

for frame in pbar.tqdm(range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1), desc="Rendering frames"):
    bpy.context.scene.frame_set(frame)
    bpy.context.scene.render.filepath = path_output + "temp.png"
    bpy.ops.render.render(write_still=True)
    img = cv2.imread(filename=path_output + "temp.png")

    result = get_screen_positions(ball)
    if result:
        # temp -> draw bounding box
        x, y = result
        r = 30
        top_left = (int(x - r), int(y - r))
        bot_right = (int(x + r), int(y + r))
        cv2.rectangle(img, top_left, bot_right, (0, 255, 0), 2)

    video.write(img)

video.release()