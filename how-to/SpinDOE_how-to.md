# How to set up Cameras and SpinDOE
Find all necessary files in /spindoe folder

## Installing dependencies
Camera tool for taking spinDOE ready images using a frame-based camera with a Lightsource mounted behind the camera

- Install all ros2 dependencies to work for tt_tracking (Thomas Gossard GitLab) (https://gitlab.cs.uni-tuebingen.de/robots/ros2-table-tennis-robot/tt_tracking)
- Source ros bash script (install/setup.bash)


## Setting up SpinDOE
Set up the actual spinDOE pipeline to estimate the spins. Original code: https://github.com/cogsys-tuebingen/spindoe

- Download SpinDOE Github and make changes according to spindoe_backup.zip
- Replace checkpoint to be epoch=6-step=15981.ckpt   (better model)


## Capturing images
Capture images using the ros2 capture pipeline

- Start camera: ros2 launch tt_spin capture.launch.py (Format needs to be changed in SpinView to 1920x1200)
- Start Recording:  ros2 run tt_spin spin_estimator_main 

- Note: Bright lightsource needs to be installed behind camera to make the dots visible

- The script detects shots and saves all images into ~/Pictures/spin_est
- Some paths may need to be changed from the GitLab code (not sure)


## Estimate spins using spinDOE


- Execute spindoe.py or doe.py (Both might work)
- Make sure, the filenames contain the correct timestamps