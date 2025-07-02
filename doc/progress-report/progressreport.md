---
author: Levin Kolmar
title: "Learning-based spin estimation of table tennis ball with an event camera"
subtitle: "Progress report week 6"
institute: "Cognitive Systems, University of Tübingen"
documentclass: scrartcl
papersize: a4
---

# Summary

Collected first dataset:

* Including 1847 simulations of different top- and backspins
* Rotation axis including sidespin of +- 20°
* Spin speed ranging between 0 and 80 rps
* Initial rotation of the ball is chosen randomly, but in a way that the logo is always visible
* Simulation is fixed to 2 rotations and the length is therefore based on the spin-speed
* 13.1 GB of data simulated in ca. 37 hours using two instances on one machine.
* ca. 50% of disk space is the frame based video which isnt really necessary

Began working on spin prediction pipeline


# Open questions
* Is there a machine that i can use for simulations like that? Its cpu restricted and i could use one instance per core. But i dont want to steal performance from a computer that is used by someone else.

* @David can you have a look at the hdf5-file, if it fits the specifications?

* What input representation would be the best to use for the network? I think voxel-grids would be a good starting point. EROS would be the other option right? 
If using voxel-grids, how to handle multiple events per pixel? Create a timesurface? Event-histogram? Or just put in raw events?

* As simulation is fixed to two rotations, it would be good to cut off a random part of the simulation length right? (not more than 50%)

* How to handle time-binning? Fixed number of bins and put avg. timestamps as second input to network? Fix by timewindow and then pad with empty bins?

* Should i hand-select a test set to inlcude a good distribution of rotations or use a recording from max or create my own (real data)?

* How does the ground-truth generation work for recorded data? what values will it produce? (Rotation axis?, spinner settings?)


# Next steps

* Work on preprocessing for simulation data (cut out ROI, filter translation events)
* Label data
* Train simple model on topspin dataset
* Train simple model (regression)
* Simulate full rotation dataset