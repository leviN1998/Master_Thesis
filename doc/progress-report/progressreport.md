---
author: Levin Kolmar
title: "Learning-based spin estimation of table tennis ball with an event camera"
subtitle: "Progress report week 16"
institute: "Cognitive Systems, University of Tübingen"
documentclass: scrartcl
papersize: a4
---

# Summary

* Encountered some problems: 
    - Simulation takes very long with new settings
    - Real data is very bad (below 10% with very good logo)
    - I tried learning with small subset that was already available, but not enough data and the ral data is not good enough
<br>

* Did some writing
* Thesis structure:
    - Introduction (Motivation of table-tennis event-cameras, simulation and learning)
    - Background (Mathematical backround and basics, Rotaiton representations, GRU, what else??)
    - Realted work (Summarize Papers but short, Firenet, event-simulator, SpinDOE, Hitchhiking rotations etc.)
    - Methodology:
        - Dataset (Simulator, Real)
        - Network (Classification, Regression)
    - Experiments (explain challenges and design decisions here or in methodology??):
        - Real-Data collection
        - Different sim setups -> move to methodolgy?
        - Training with first (bad) Dataset
        - Training with better dataset (->now)
        - Training regression with that dataset
        - including real data into training and finetune
        - Try other models
    - Conclusion

<br>

* Implemented rotation representation as suggested by paper (Hitchhiking rotations)
* Started with collection of real data (again)


# Open questions

* What do you think about the structure? How to decide if topic belongs to experiment or method?
* Do you think it would work to have a moving camera in blender, that only records a small ROI with the ball? This would improve sim-time significantly
* Would it make sense to only simulate e.g. 2 rotations and copy them togehter to match the full duration? Video lenght is about 0.4s with 130 rps that is 52 rotations.
* I recorded real data with 3 different bias settings. Can we have a look together and decide if it is good enough to record more data like that or in which direction to tune? (One of the videos in discord)
* How important would you say that sidespins are? Because collecting data for top-/backspins with +-20 degrees sidespin makes it easier to have the logo visible. (If it is visible it will always be visible). Also 70% of the ball-gun settings produce these spins.

# Next steps

* Continue writing
* record more real data
* make simulation more efficient
* train with new data (classification)
* check trained model with real data
* Train regression
* check with real data
* Do finetuning with real data