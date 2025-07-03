# Summaries of useful papers

Here i want to add my summaries of all papers that i had to read.

## Bachelorarbeit Schnitt
* Voxel Grids as input representation (Timesurface second option)
* binary cross entropy loss used

## Hitchhiking Rotations

## ICNS Simulator

## Hydra template
https://github.com/ashleve/lightning-hydra-template

## Spin Detection in Robotic Table Tennis (Tebbe)
* 3 different aproaches, background-difference to find logo. CNN to predict orientation of Logo. Evaluate trajectory and derife from magnus force (highest accuracy)
* For CNN, smallest ResNet architecture used. Additional two fully connected (512) at the end
* axis-angle representation or Quaternions used.
* Also classify if the logo is even visible
* Look up geodesic distance in SO(3) for the loss with regression


## SpinDOE (Gossard)
* CNN to localize the dots then identify with geometric hashing
* Has a dataset -> maybe convert to ev and train on
* Uses dotted Balls
* Think about orientation aproach -> multiple pipeline steps (Very framebased thinking)
* Uses a custom ball spinner for ground truth generation


## Table tennis ball spin estimation with an event camera (Gossard)
* Timesurface -> Optical flow estimation is used. Error: (10.7 +- 17.3) rps and (32.9 +- 38.2) °
* TOS and EROS for better timesurfaces.
* pipeline: Ball tracker -> Extract Logo events -> Estimate Spin
* Kalman filter to estimate velocity and radius
* Extract logo events (Trash edges)
* -> Understand Spin estimation part
* ball spinner and ball thrower used as ground truth


## Event-based Ball Spin Estimation in Sports (Nakabayashi)
* TODO


## FireNet (Scheerlinck)
* Used for video reconstruction
* Inputs are temporal bins. Each event cotributes its polarity to the two closest temporal bins.
* More precise:
    Inputs are splitted into frames (sames as a video would be)
    Those frames are the inputs for different recurrent steps
    Every input is divided into n (5 i think) time bins to form the voxel grid
    This is the channel dimension of the network input
* Architecture with GRU cells


## Learning to Detect Objects with a 1 Megapixel Event Camera (Perot)
* I think the firenet model is better suited than this one
* Detailed description of event-representation and differenc choices
* Uses end to end aproach -> Events are used directly instead of converting them into grayscale
* Different input representations: Histograms of events, Time Surfaces, Event Volumes.



## Detecting Stable Keypoints from Events through Image Gradient Prediction (Chiberre)
* TODO


## Event-based Asynchronous Sparse Convolutional Networks (Messikommer)
* TODO


## FARSE-CNN: Fully Asynchronous, Recurrent and Sparse Event-based CNN
* TODO


## Recurrent Vision Transformers for Object detection with event cameras
* TODO


## GG-SSMs: Graph-Generating State Space Models
* TODO


## Timo Stoff
* Some very useful event stuff (data-augmentation, representation etc.)