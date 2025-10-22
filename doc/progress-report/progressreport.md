---
author: Levin Kolmar
title: "Learning-based spin estimation of table tennis ball with an event camera"
subtitle: "Progress report week 20"
institute: "Cognitive Systems, University of Tübingen"
documentclass: scrartcl
papersize: a4
---

# Summary

* Found a big bug in preprocessing, the simulated datasets only had a 1/4 of the ball visible.
* Tested Accuracy on real data (See confusion matrix). Most of the missclassifications either have the correct speed or spin direction. 
The highest probability of failing comes with high rps values or high siedespin settings.
* Trained regression model
* Trained model on Dotted balls
* Recorded dataset with less sidespin values
* Almost finished writing of chapter Methodology

![Confusion matrix for real data](../conf_matrix2.png){ width=75% }




# Open questions



# Next steps

* Check regression performance
* Do last experiments
* Finish methodolgy chapter
* Do more writing