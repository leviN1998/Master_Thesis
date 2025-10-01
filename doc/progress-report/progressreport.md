---
author: Levin Kolmar
title: "Learning-based spin estimation of table tennis ball with an event camera"
subtitle: "Progress report week 17"
institute: "Cognitive Systems, University of Tübingen"
documentclass: scrartcl
papersize: a4
---

# Summary

* Finished Simulation of new dataset with 13 000 top- and backspins with rps between 10 and 140. 
* Preprocessed new dataset
* Training on that dataset (over night, hopefully results tomorrow)
* Almost finished recording real dataset with 10 recordings per ball-gun setting. The recordings are hand-picked to have the logo visible on the table tennis ball. (~30% of shots)
* Continued coding for regression
* Continued writing.


# Open questions

* At the moment one trajectory has about 30 to 50 rotations. Could it be feasible to only simulate 5 and then just repeat the video with different noise?

# Next steps

* Look at results
* Finish real dataset
* Finetune with real data
* Train regression
* Try other model architectures
* Continue writing