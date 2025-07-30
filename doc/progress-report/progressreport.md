---
author: Levin Kolmar
title: "Learning-based spin estimation of table tennis ball with an event camera"
subtitle: "Progress report week 10"
institute: "Cognitive Systems, University of Tübingen"
documentclass: scrartcl
papersize: a4
---

# Summary

* Dataloading almost working (had a lot of bugs there)
* First training runs with firenet on the simulated dataset but not working due to bugs in dataloading


# Open questions
* Do we have a calibration library working to transform a spin measured by the frame-camera into the event camera?
* Which event camera should i use for data collection?
* David: do you have code to create the metavision-hdf5 format from the one we use?

# Next steps

* Finish dataloading
* Train Firenet on Top-/Backpsin dataset
* Train Firenet on Full-spin dataset
* Collect real data to check if model works on that as well
* Move to regression instead of classification