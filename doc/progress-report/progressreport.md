---
author: Levin Kolmar
title: "Learning-based spin estimation of table tennis ball with an event camera"
subtitle: "Progress report week 9"
institute: "Cognitive Systems, University of Tübingen"
documentclass: scrartcl
papersize: a4
---

# Summary

* Not that much, had an exam today
* Implemented FireNet and trained succesfully on n-mnist
* Some code fixes on the simulator
* Image collecting and SpinDOE now fully working
* Labeled and preprocessed own dataset


# Open questions
* When using Firenet i needed to have residual connections around every GRU cell to be able to learn when including full noise timesteps. Is is okay or do i need to bugfix somewhere?

# Next steps

* Finish DataLoader class for own dataset
* Train Firenet on Top-/Backpsin dataset
* Train Firenet on Full-spin dataset
* Collect real data to check if model works on that as well
* Move to regression instead of classification