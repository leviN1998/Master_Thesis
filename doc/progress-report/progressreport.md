---
author: Levin Kolmar
title: "Learning-based spin estimation of table tennis ball with an event camera"
subtitle: "Progress report week 14"
institute: "Cognitive Systems, University of Tübingen"
documentclass: scrartcl
papersize: a4
---

# Summary

* SpinDOE dataset is now fully preprocessed and usable
* The dataset contains 2 recordings for every (143) ball gun setting. For every setting, the ground truth is measured using SpinDOE. 
Every trajectory was cut out of the event-recording and the ROI is extracted using hand-labeled positions.
* Labeled SpinDOE dataset into 6 labels (top-/backspin) (slow, mid, fast). Real data has rps up to 135rps and simulation only 80rps.
* tested model on real data. Very bad results see image.

![Confusion matrix \label{fig:conf}](../conf_matrix.png){ width=50% }

![Real vs sim Data](../sim_vs_real.png){ width=75% }

* Main problem: sim data has trahectory from right to left and real the other way around. I will try to flip and train a new model until tomorrow

# Open questions

* How could the data be improved? trajectory instead of straight line? faster/slower ball? different scale?
* Look at examples of sim vs real together.


# Next steps

* improve simualtion
* train new model