# Drone-Detection-on-Acoustic-Spectrograms

Unimodal script:

This Python script captures a video stream, detects drones using a machine learning model, and streams the processed video along with logging results via a socket to a fusion component. It also outputs detection details in real-time.


Fusion script:

This Python script processes and synchronizes acoustic data from multiple sources (GFAI and Respeaker) and performs fusion to detect drones. It streams video with detection results and sends JSON logs to a platform.
