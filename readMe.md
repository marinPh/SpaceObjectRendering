# Rendering Objects in an Orbit our Earth

## Introduction

This project was directed by the EPFL Computer Vision lab that wanted a way to create many images and trajectories. Many of blender were calculated in previous projects. The task during the semester was create from many code blocks and concepts a way to create usable trajectories of objects to train a computer vision model. > DISCLAIMER: This code is not my IP as it was written with the supervision of my lab and I am simply allowed to show my work.

## Structure

In c_root you will find the different scripts used by the bashc commands and the 3 bash commands to learn how to use them simply add the -h flag after the command

## Usage

### Setup

Firstly you must create the directory c_root/blend and add your individual objects. Then the code will manage the settings at the matrix stage.

### Matrix

The matrix.sh command will calculate the inertia matrix given the object is uniformly dense and apply the parameters to the object. The very interesting fact is that was that the algorithm was parallelised by using pymeshlab that decomposed the object in points uniformly and using numpy calculated the center of mass and the inertia matrix

### Pose and Trajectory creation

This step was the most tedious as the requirements were hard to apply. The solution was to create a 3D representation of the trajectory in the camera frustum. The trajectories are always 100 frames, we kept the initial position and direction uniformly random and the speed was adjusted to complete 100 frames, the interesting part is seeing visually the points of the objects in space before rendering them.

### Rendering

We use the parameters and definition of our poses to create sequent images (or randomly depending of pose creation parameters).

## State of project

Project is finished but after taking a look recently some modifictions would be needed to be used by OutOfLab users so I will implement the settings extractor as the oringinal blender is too big for git.

## Examples and Report 

You can find examples images in ./examples and example htmls of objects used in ./objects/html/ and to find the htmls of trajectories ./input/objectID_poseID (000 is random so no trajectories) and their will be my report as a PDF if you are interested.
