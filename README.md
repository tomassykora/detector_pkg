# Objects finder
This repository offers source codes and ROS configuration files to run a robot capable of navigation, objects detection and recognition.

## Detector_pkg
Ros package for object detection and recognition.

### Package description

Package contains a node which can detect objects in front of a sensor. By using tensorflow recognition node from image_recognition package (tue-robotics) tries to recognize trained objects.

### How to

Before running detector_pkg, train neural network like in this tutorial [image_recognition](https://github.com/tue-robotics/image_recognition). Then start the tensorflow node:

        rosrun tensorflow_ros object_recognition_node _graph_path:=<path/to/the/graph/from/training/output_graph.pb> _labels_path:=<path/to/the/labels/from/training/output_labels.txt>

Run detector node:

        rosrun detector_pkg detector

## Kobuki_exploration
ROS package implementing actionlib client to frontier_exploration server. Needed frontier_exploration launch file is available in the [launch](https://github.com/tomassykora/detector_pkg/tree/master/launch) directory.

## Launch files
All necessary launch files to run navigation stack and whole object finder demo are available in the [launch](https://github.com/tomassykora/detector_pkg/tree/master/launch) directory.
