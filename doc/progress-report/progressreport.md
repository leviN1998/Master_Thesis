---
author: Levin Kolmar
title: "Learning-based spin estimation of table tennis ball with an event camera"
subtitle: "Progress report week 19"
institute: "Cognitive Systems, University of Tübingen"
documentclass: scrartcl
papersize: a4
---

# Summary

* Writing
* Training 2 class (topspin / backspin) training: 98% accuracy. Testing on real data: 93% accuracy
* Training 6 class (+ fast, mid, slow) training: 97% accuracy. Testing on real data: 85% accuracy. (only 90% on train set. Prbably wrong ball-gun labels)
* Finished regression code, but bug on network head that prevents good learning.

# Open questions

* What can i do to improve performance on real data?
* Do you have a good master-thesis to take a look how the writing style etc. should be?
* Would it be good to record a smaller real-data set, where the ball-gun cant mess up labels? E.g. spin setting without sidespin.
* What ball-gun settings did you use in the spin-estimation paper? No sidespins right?
* Should i try tocompare against dotted balls? (I have a simulated dataset for that aswell)


# Next steps

* Bugfix regression
* Train ergression
* record smaller dataset
* Find setups for better results
* Maybe train on dotted balls
* Continue writing