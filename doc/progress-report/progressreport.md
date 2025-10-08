---
author: Levin Kolmar
title: "Learning-based spin estimation of table tennis ball with an event camera"
subtitle: "Progress report week 18"
institute: "Cognitive Systems, University of Tübingen"
documentclass: scrartcl
papersize: a4
---

# Summary

* More writing
* Finished recording real data (10 recordings per ball-gun setting) 560 recordings in total
* Preprocessed real data with Davids convolution aproach: worked very good.
* Trained top vs backspin and 6 classes with also fast, mid slow on simulated data. (Accuracy ~0.98)
* Validated both on real data with 23% and 12.5% accuracy
* Finetune on real data overnight

# Open questions

* What can i do to improve the performance on real data? Including real and simulated data into training at the same time would be my last idea for now.


# Next steps

* Look at results
* Train regression
* Try other model architectures
* Continue writing