---
author: Levin Kolmar
title: "Learning-based spin estimation of table tennis ball with an event camera"
subtitle: "Progress report week 1"
institute: "Cognitive Systems, University of Tübingen"
documentclass: scrartcl
papersize: a4
---

# Summary

More work on the simulator: <br>

* Created table tennis ball with logo in Blender (\autoref{fig:image})
* Ball spinning animation now working with spins that can be randomly generated.
* Created animation that includes ball physics. A realistic ball that can be shot and is affected by gravity. Probably overkill, just using a ball that moves in a line would be enough i think.
* Spinning ball with the creation of the events still has some bugs until i figure out how to properly set all the parameters of the simulated sensor.

![Table tennis ball with andro logo\label{fig:image}](../ball.png){ width=75% }

# New ideas

# Open questions

* What are good values for max-spin? In maxs Thesis he used 0-20 rps in the simulator, but for the real data it was faster.

# Next steps
* Fix bugs of the event rendering
* Align parameters with specification of real world, image-size, ball size in pixels etc.
* Implement ground truth saving (ball-position and rotation). To be able to cut the ball-region out of the image / event-stream.
* Create first dataset with simple ball rotation
* Implement ball movement
* Create dataset with moving ball



# Bibliography
