---
author: Levin Kolmar
title: "Learning-based spin estimation of table tennis ball with an event camera"
subtitle: "Progress report week 5"
institute: "Cognitive Systems, University of Tübingen"
documentclass: scrartcl
papersize: a4
---

# Summary

More work on the simulator: <br>

* Some changes to the simulation, including faster spins
* started tuning simulator parameters: (on-threshold, off-threshold, noise-threshod, latency, time-constant, temporal jitter, background-noise-rate, refractory-period)
* Not entirely happy with the resolution of the logo in the event-video
* If that problem is fixed, simulator should be ready to generate first dataset with ~2000 top- and backspins in approx. 80 hours if using only one machine.


# Open questions

* How could the visiblility of the logo and therefore the visibility of the spin be increased? Moving the camera closer to the ball has the downside that we said the most realistic size for the ball is ~50 pix which it is at the moment. Reducing the the roation speed would also help, because with 20 rps instead of 80 the timeframes are longer and we get more events. 
* Do you see other problems in the video? I wnat to fix everything possible before spending 2days on two machines creating the dataset.

# Next steps

* Increase logo-resolution
* Tune with real recordings
* Change hdf5 format to fit guidelines
* Simulate dataset
* Start with first model learning on the simple dataset