---
author: Levin Kolmar
title: "Learning-based spin estimation of table tennis ball with an event camera"
subtitle: "Progress report week 3"
institute: "Cognitive Systems, University of Tübingen"
documentclass: scrartcl
papersize: a4
---

# Summary

More work on the simulator: <br>

* The simulator is now working more or less as visible in the videos on discord. <br>

A simulation is created like this: total ammount of frames can be set to  e.g. 500 frames. Rotations per second can be set to eg 40 rps. The simulation
will contain two rotations, therefore the simulation length is total_rot / rps. The fps are total_frames / sim_length. The corresponding video will have a different fps, because in real-time this would be too short. The simulation can be saved as video, hdf5 file and event video. The ground-truth-file contains 
the rotation axis, rps and the per-frame screen-coords of the ball. (There is a nasty bug to fix).


# New ideas

# Open questions

* Framerates are just for blender to generate images where the event-simulator gets timestamps to interpolate. What would a good framerate be? 
I recorded simulations for framerates of [300, 500, 1000, 1500] fps. The smallest one takes about 1:30 minutes to generate. There could be some improvements.

* Is the way how the simulation works now a good idea? Fixed amount of frames and total spins, and the rest will adapt to those values.

* What infomation should be inside the hdf5 file? Now it contains one array: events: [p, x, y, t]. 

* Is the ball-speed even relevant? As long as the ball covers some distance.

* What do you think of the size of the event videos in discord. Should the ball be smaller? Should it be smaller in terms of pixels? I could make it smaller,
but zoom in, then it would cover less pixels.

* Can we check if the spin generation is correct? (David)

# Next steps

* Fix Bugs for position extraction.
* Adjust bias settings in the simulator.
* Find configuration for first dataset.
* Simulate dataset

# Bibliography
