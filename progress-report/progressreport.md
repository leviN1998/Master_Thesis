---
author: Levin Kolmar
title: "Learning-based spin estimation of table tennis ball with an event camera"
subtitle: "Progress report week 4"
institute: "Cognitive Systems, University of Tübingen"
documentclass: scrartcl
papersize: a4
---

# Summary

More work on the simulator: <br>

* Some changes to the simulation, as discussed in Discord <br>
* Dataset generation picks values out of a table. Therefore process can be split to multiple machines and can also be paused/restarted.
* Fist Dataset should only include top- and backspins in a range of 5 - 80 rps. See questions.



# Open questions

* How much variation in the topspins should be included? At the moment vectors that have an angle < 20° to the horizontal axis are selected.
* Initial Orientation of the ball should be random right? There are the options: 1. facing the camera. 2. totally random. Evenly spaced, eg. 10 different orientations per sample.
* How to save ground-truth rotations. Relative to what? Easiest way would be relative to world-coordinate and save the ball direction, camera position and angle.
* How many labels would be good for the basic dataset? [top, back]? 
* Does the rps spacing make sense this way? Spins and rps are creted using points in a cube, as discussed last week. The resulting rps when filtering a cube with 40 points per edge are: [10, 14, 18, 19, 22, 23, 26, 27, 28, 30, 31, 32, 34, 35, 36, 39, 40, 43, 44, 45, 47, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]. The distribution gets denser with increasing rps (this makes sense), but for the small numbers this effect is very strong. 



# Next steps

* Finish code for dataset creatin (almost done)
* Fix details discussed this week
* Simulate dataset
* Start with first model learning on the simple dataset