# Detector_pkg
Ros package for object detection and recognition.

## Package description

Package contains a node which can detect objects in front of a sensor. By using tensorflow recognition node from image_recognition package (tue-robotics) tries to recognize trained objects.

## How to

Before running detector_pkg, train neural network like in this tutorial [image_recognition](https://github.com/tue-robotics/image_recognition). Then start the tensorflow node:

        rosrun tensorflow_ros object_recognition_node _graph_path:=<path/to/the/graph/from/training/output_graph.pb> _labels_path:=<path/to/the/labels/from/training/output_labels.txt>

Run detector node:

        rosrun detector_pkg detector

## Installation

Clone the repo in your catkin_ws:

        cd ~/catkin_ws/src
        git clone https://github.com/tomassykora/detector_pkg.git

Build your catkin workspace
        cd ~/catkin_ws
        catkin_make
