# Hand Controlled UI

This is a Python project where you control on-screen UI using your hand in front of a camera. The idea is based on Iron Man’s Jarvis system, but simplified and built entirely in Python.

i used MediaPipe for hand tracking and OpenCV for camera input and rendering. Everything runs through a webcam and detects basic hand movements and their angles relative to the knuckles & palm.

Note: I recommend using Python 3.11 for this. Does not work with Python versions 3.12+.
## Overview

The program tracks your hand in real time and uses finger positions to interact with visual UI elements. The interface is drawn directly on the camera feed. There is no mouse or keyboard interaction once the program is running.

## How It Works

OpenCV captures frames from the webcam and draws the UI.  
MediaPipe detects hand landmarks and returns their positions.  
The landmark positions are processed to detect gestures.  
Gestures are mapped to UI actions.

The system is simple and direct. There is no machine learning beyond MediaPipe’s built-in model.

## Technologies Used

i used Python for the entire project.
OpenCV handles video input and rendering.  
MediaPipe Hands provides hand and finger tracking.  
NumPy is used for basic calculations.

## Project Goal

Experimenting with visual design & control
This project is meant as a learning project, not a finished product.
