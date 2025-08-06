---
author: Levin Kolmar
title: "Learning-based spin estimation of table tennis ball with an event camera"
subtitle: "Progress report week 10"
institute: "Cognitive Systems, University of Tübingen"
documentclass: scrartcl
papersize: a4
---

# Summary

* Fixed dataloading problems
* Trained Firenet on the dataset with top- and backspins. It had 6 classes: top or back and 3 idfferent speeds.  
  Test accuracy was 0.94. \url{https://api.wandb.ai/links/levin-kolmar-master/u1nrmuoo}
* I found out that the duration for loading a batch takes 2x as long as a forward pass. I could pre-compute the transformations (voxel-grid).


# Open questions
* Is 94% accuracy on the test set enough for 6 classes (Doesn't sound that good)
* What is the best next step? Include data-augmentations, benchmark with real data, move on to regression, train on more difficult datasets with more spin-sttings (side-spin included). 
* Should i try to include different ball-sizes and test how well that works?
* De we have something to extract the ball ROI out of the event-data?


# Next steps

* Setup cameras and calibration for real data collection
* Collect real data
* Move on to regression
* Move on to sidespin
* Include usefull data-augmentations
* Change Simulations to be more realistic and harder. Lighting, ball-size etc.