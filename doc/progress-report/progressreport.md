---
author: Levin Kolmar
title: "Learning-based spin estimation of table tennis ball with an event camera"
subtitle: "Progress report week 7"
institute: "Cognitive Systems, University of Tübingen"
documentclass: scrartcl
papersize: a4
---

# Summary

* Recorded new dataset with simulations looking closer to Maxs data 
* Preprocessing: Creating Voxel Grids and extracting ROI (See video)
* Started implementation of FireNet as Baseline


# Open questions
* Would it be better to have the whole Dataset in one big hdf5 file to exploit its loading capabilities or should i load every simulation from its own hdf5 file?

* Are there preprocessing steps that you would recommend? Filter out noise etc.?

* Did i get this right, that the temporal input should be 2 dimensional when using voxel grids? For one sequence we would have n "frames" (recurrent part).
Then for every frame there are time-bins that are used as channel input. So the time-bins are fixed number but "frames" not. At least thats how i understood the Firenet paper. This is also the difference to Maxs implementation.

# Next steps

* Label dataset
* Finish Firenet Code
* Train Firenet
* Move on to regression instead of Classification