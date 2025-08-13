---
author: Levin Kolmar
title: "Learning-based spin estimation of table tennis ball with an event camera"
subtitle: "Progress report week 10"
institute: "Cognitive Systems, University of Tübingen"
documentclass: scrartcl
papersize: a4
---

# Summary

* Updated simulator to handle different trajectories and be more flexible.
* Started working on regression-code
* Working on real-data collection
*         TP  - TN  -  FP  -  FN
Class 0  0     138.6  0.9    2.9      (topspin slow) \newline
Class 1  15.7  123.4  2.3    1.0      (topspin mid) \newline
Class 2  52.6  88.8   1.0    0        (topspin fast) \newline
Class 3  2.6   137.0  0.6    2.2      (backspin slow) \newline
Class 4  18.8  120.1  3.6    0        (backspin mid) \newline
Class 5  44.4  95.8   0      2.2      (backspin fast) \newline

# Open questions
* Is it a problem, that the class balance in the test set is very bad?
* It would be more interesting to find out which class get missclassified to which other class right? (Confusion matrix with all classes)

# Next steps

* Collect real data
* Compare with real data
* Move on to regression
* Move on to sidespin
* Include usefull data-augmentations
* Change Simulations to be more realistic and harder. Lighting, ball-size etc.