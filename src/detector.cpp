/*
 * Real-time object detector and recognizer
 * The detector process point cloud data from a 3d senzor 
 * and determines whether there are some objects in its view field or not.
 * If there are objects, calls a service of tensorflow_ros pkg, which 
 * determines whether found objects are known objects.
 *
 * School project 2016-2017
 * Author: Tomas Sykora, xsykor25@stud.fit.vutbr.cz
 */

#include <iostream>
#include <cmath>
#include <sstream>
#include <vector>
#include <time.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl_conversions/pcl_conversions.h>

#include "ros/ros.h"
#include <tf/tf.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Twist.h>
#include "geometry_msgs/PointStamped.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_recognition_msgs/Recognize.h>
#include <image_geometry/pinhole_camera_model.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>

#define UNKNOWN_PROB_TRESHHOLD 0.96

typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

using namespace message_filters;
namespace enc = sensor_msgs::image_encodings;

int first_time = 0;
ros::Time actual_time;

ros::Publisher image_pub;
ros::Publisher objects_pub;
ros::Publisher velocity_pub;

image_geometry::PinholeCameraModel cam_model;

bool start_manipulating = false;
int file_num = -1;

sensor_msgs::Image imageCb(const sensor_msgs::ImageConstPtr& msg, int centroid_x, int centroid_y)
{
  cv_bridge::CvImageConstPtr cv_ptr;
  cv_bridge::CvImage img_roi_output;

  try
  {
    cv_ptr = cv_bridge::toCvShare(msg, enc::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }

  int x_offset, y_offset;

  if (centroid_x-30 > 0)
    x_offset = centroid_x-30;
  else
    x_offset = 0;

  if (centroid_y-5 > 0)
    y_offset = centroid_y-5;
  else
    y_offset = 0;

  int rect_x, rect_y;

  if (x_offset+60 > 639)
    rect_x = 639-x_offset;
  else
    rect_x = 60;

  if (y_offset+110 > 479)
    rect_y = 479-y_offset;
  else
    rect_y = 110;

  /*if (centroid_x-30 > 0)
    x_offset = centroid_x-30;
  else
    x_offset = 0;

  if (centroid_y-50 > 0)
    y_offset = centroid_y-50;
  else
    y_offset = 0;

  int rect_x, rect_y;

  if (x_offset+85 > 639)
    rect_x = 639-x_offset;
  else
    rect_x = 85;

  if (y_offset+120 > 479)
    rect_y = 479-y_offset;
  else
    rect_y = 120;*/

  cv::Rect roi(x_offset, y_offset, rect_x, rect_y);
  img_roi_output.header = msg->header;
  img_roi_output.encoding = enc::BGR8;
  img_roi_output.image = cv_ptr->image(roi);

  file_num = random();
  std::ostringstream name;
  name << "file" << file_num << ".jpg";
  imwrite(name.str(), img_roi_output.image);

  sensor_msgs::ImagePtr ros_msg_ptr = img_roi_output.toImageMsg(); 

  return *ros_msg_ptr;
}


Eigen::Affine3f get_matrix(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float *z_after_rotation)
{
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;

  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.01);
  seg.setInputCloud (cloud);
  seg.segment (*inliers, *coefficients);

  if (inliers->indices.size () == 0)
  {
    PCL_ERROR ("Could not estimate a planar model for the given dataset.\n");
    return Eigen::Affine3f::Identity();
  }

  Eigen::Matrix<float, 1, 3> floor_plane_normal_vector, xy_plane_normal_vector, rotation_vector;

  // Get normal vector of the floor plane
  floor_plane_normal_vector[0] = coefficients->values[0];
  floor_plane_normal_vector[1] = coefficients->values[1];

  // Check orientation of the normal
  if (coefficients->values[2] > 0)
    floor_plane_normal_vector[2] = coefficients->values[2];
  else
    floor_plane_normal_vector[2] = - coefficients->values[2];

  // Get normal vector of the XY plane
  xy_plane_normal_vector[0] = 0.0;
  xy_plane_normal_vector[1] = 0.0;
  xy_plane_normal_vector[2] = 1.0;

  // Get angle between XY plane normal and floor plane normal
  float theta = acos((floor_plane_normal_vector[0] * xy_plane_normal_vector[0] + 
                      floor_plane_normal_vector[1] * xy_plane_normal_vector[1] + 
                      floor_plane_normal_vector[2] * xy_plane_normal_vector[2]) 
                      / (sqrt(powf(floor_plane_normal_vector[0], 2) + 
                              powf(floor_plane_normal_vector[1], 2) + 
                              powf(floor_plane_normal_vector[2], 2)) 
                        * sqrt(powf(xy_plane_normal_vector[0], 2) + 
                               powf(xy_plane_normal_vector[1], 2) + 
                               powf(xy_plane_normal_vector[2], 2))));

  // Calculate transform matrix
  Eigen::Affine3f transform_matrix = Eigen::Affine3f::Identity();

  // Calculate 'z' coordinate value of cross point of original plane and z axis
  float original_z_coordinate = -(coefficients->values[3] / coefficients->values[2]);

  // Calculate 'z' distance between rotated plane and XY plane
  *z_after_rotation = original_z_coordinate * cos(theta);

  // Define rotation
  transform_matrix.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitX()));

  return transform_matrix;
}

int getIndex(float x, float y, float z, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  double dist = 100.0;
  int index = 0;
  double tmp_dist;

  ROS_INFO_STREAM("Finding most similar coords to: " << x << " " << y << " " << z);

  for (int i = 0; i < 480; i++)
  {
    for (int j = 0; j < 640; j++)
    {
      //if (!isnan(cloud->points[i*640+j].x) && !isnan(cloud->points[i*640+j].y) && !isnan(cloud->points[i*640+j].z))
      if (isFinite(cloud->points[i*640+j]))
      {
        tmp_dist = sqrt(powf((cloud->points[i*640+j].x-x),2) + 
                        powf((cloud->points[i*640+j].y-y),2) + 
                        powf((cloud->points[i*640+j].z-z-0.6),2)); /* z -0.1 is a heuristic to get rid of floor points) */
      }

      if (tmp_dist < dist) 
      {
        index = i * 640 + j;
        dist = tmp_dist; 
      }
    }
  }

  ROS_INFO_STREAM("Using point: " << cloud->points[index].x << ", " << cloud->points[index].y << ", " << cloud->points[index].z);

  return index;
}

int computeCentroidIndex(const std::vector<int> &indices)
{
  int centroid_index = 0;

  for (size_t i = 0; i < indices.size (); ++i)
  {
    centroid_index += indices[i];
  }

  return (int)centroid_index/indices.size();
}

void object_detector(const sensor_msgs::PointCloud2ConstPtr& input, const sensor_msgs::ImageConstPtr& image, const sensor_msgs::CameraInfoConstPtr& info_msg, ros::ServiceClient &client, ros::NodeHandle &n, MoveBaseClient &ac)
//void object_detector(const sensor_msgs::ImageConstPtr& image, ros::ServiceClient &client, ros::NodeHandle &n)
//void object_detector(const sensor_msgs::PointCloud2ConstPtr& input, ros::ServiceClient &client, ros::NodeHandle &n)
{
  ROS_INFO_STREAM("Handle pointcloud...");

  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg (*input, *input_cloud);
  float z_after_rotation;
  static Eigen::Affine3f transform_matrix = Eigen::Affine3f::Identity();

  if (first_time == 0)
  {
    cam_model.fromCameraInfo(info_msg);
    actual_time = ros::Time::now();
    first_time = 1;
    transform_matrix = get_matrix(input_cloud, &z_after_rotation);
  }

  // Select region of interest from point cloud by filering some points
  pcl::PassThrough<pcl::PointXYZ> pass_window;
  pass_window.setInputCloud (input_cloud);
  pass_window.setFilterFieldName ("z");
  pass_window.setFilterLimits (0.0, 2.0);
  //pass.setFilterLimitsNegative (true);
  pass_window.filter (*cloud);

  /*pass_window.setInputCloud (cloud);
  pass_window.setFilterFieldName ("x");
  pass_window.setFilterLimits (-0.85, 0.85);
  //pass_window.setFilterLimitsNegative (true);
  pass_window.filter (*cloud);*/

  pass_window.setInputCloud (cloud);
  pass_window.setFilterFieldName ("y");
  pass_window.setFilterLimits (-0.1, 0.3);
  //pass_window.setFilterLimitsNegative (true);
  pass_window.filter (*cloud);

  // Commenting this for the demo video
  if (ros::Time::now() - actual_time > (ros::Duration)(30))
  {
    actual_time = ros::Time::now();
    transform_matrix = get_matrix(cloud, &z_after_rotation);
    ROS_INFO_STREAM("Segmentation done.");
  }
 
  // Transform the point cloud (if sensor angle is 90 degrees, skip this)
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::transformPointCloud (*cloud, *transformed_cloud, transform_matrix);

  // Plane removal
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_without_plane (new pcl::PointCloud<pcl::PointXYZ> ());

  // Create the filtering object
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (transformed_cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (z_after_rotation-0.03, z_after_rotation+0.03);
  pass.setFilterLimitsNegative (true);
  pass.filter (*cloud_without_plane);


  // Find remaining objects
  pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod (tree);
  normal_estimator.setInputCloud (cloud_without_plane);
  normal_estimator.setKSearch (100);
  normal_estimator.compute (*normals);

  /*pcl::IndicesPtr indices (new std::vector <int>);
  pcl::PassThrough<pcl::PointXYZ> pass2;
  pass2.setInputCloud (cloud_without_plane);
  pass2.setFilterFieldName ("z");
  pass2.setFilterLimits (0.0, 1.0);
  pass2.filter (*indices);*/

  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize (1800);
  reg.setMaxClusterSize (10000);
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (30);
  reg.setInputCloud (cloud_without_plane);
  //reg.setIndices (indices);
  reg.setInputNormals (normals);
  reg.setSmoothnessThreshold (11.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold (10.0);

  std::vector <pcl::PointIndices> clusters;
  reg.extract (clusters); 
  
  ROS_INFO_STREAM("Found " << clusters.size() << " objects."); 

  bool found_known_object = false;
  std_msgs::Bool found_objects;
  float nav_goal_x, nav_goal_y, nav_goal_orientation;

  for (int i = 0; i < clusters.size() && !start_manipulating; i++)
  {
    ROS_INFO_STREAM("\n\n");

    Eigen::Vector4f centroid;

    pcl::compute3DCentroid (*cloud_without_plane, clusters[i], centroid);
 
    /*tf::Point tf_point; 
    tf_point[0] = centroid[0]; tf_point[1] = centroid[1]; tf_point[2] = centroid[2];
   
    tf::Transformer transformer;
     
    tf::Stamped<tf::Point> tf_point_in(tf_point, image->header.stamp, "camera_depth_optical_frame"); 
    tf::Stamped<tf::Point> tf_point_out(tf_point, image->header.stamp, "camera_rgb_optical_frame");
 
    transformer.waitForTransform("camera_rgb_optical_frame", "camera_depth_optical_frame", ros::Time::now(), ros::Duration(0.0)); 
    transformer.transformPoint("camera_rgb_optical_frame", boost::ref(tf_point_in), boost::ref(tf_point_out));
*/
    cv::Point3d p(centroid[0]-0.025, sqrt(powf(centroid[1],2)-0.000625), centroid[2]);
    cv::Point2d pixel = cam_model.project3dToPixel(p);

    ROS_INFO_STREAM("Computed centroid: " << centroid[0] << " " << centroid[1] << " " << centroid[2] << " " << centroid[3]);

    //int point_cloud_index = getIndex(centroid[0], centroid[1], centroid[2], input_cloud);
    int point_cloud_index = getIndex(centroid[0], centroid[1], centroid[2], input_cloud);
    int x_2d = point_cloud_index % 640;
    int y_2d = point_cloud_index / 640;

    ROS_INFO_STREAM("Index of centroid in cloud: " << point_cloud_index);
    ROS_INFO_STREAM("Object num. " << i << ": y coord: " << y_2d);
    ROS_INFO_STREAM("Object num. " << i << ": x coord: " << x_2d); 
    //ROS_INFO_STREAM("Cv coords: " << pixel.x << ", " << pixel.y); 
    /* don't understand why it's sometimes negative, another time positive */
    if (centroid[1] < 0)
      nav_goal_x = centroid[1] * (-1);
    else 
      nav_goal_x = centroid[1];
    nav_goal_y = centroid[0] * (-1);
    nav_goal_orientation = centroid[0] * (-1);

    std::vector<image_recognition_msgs::Recognition> recognitions;
    sensor_msgs::Image image_req = imageCb(image, x_2d, y_2d);
    //sensor_msgs::Image image_req = imageCb(image, pixel.x, pixel.y);

    ROS_INFO_STREAM("Image region info: Height: " << image_req.height << 
                                      ", Width: " << image_req.width << 
                                      ", Step: " << image_req.step << 
                                      ", Data vector size: " << image_req.data.size() <<
                                      ", file: " << file_num);

    image_recognition_msgs::Recognize srv;
    image_recognition_msgs::CategoryProbability best;

    srv.request.image = image_req;
    best.label = "unknown";
    best.probability = UNKNOWN_PROB_TRESHHOLD;

    /* Pause exploration, because recognizing takes a long time (seconds). */
    ac.cancelAllGoals();
    found_objects.data = true;
    objects_pub.publish(found_objects);

    // Try to recognize known objects
    ROS_INFO_STREAM("Starting recognition.");

    if (client.call(srv)) 
    {
      recognitions = srv.response.recognitions;

      for(std::vector<image_recognition_msgs::Recognition>::iterator i = recognitions.begin(); i != recognitions.end(); ++i) 
      {
        best.label = "unknown";
	//best.probability = i->categorical_distribution.unknown_probability;
        best.probability = UNKNOWN_PROB_TRESHHOLD;

	for (unsigned int j = 0; j < i->categorical_distribution.probabilities.size(); j++) 
        {
	  if (i->categorical_distribution.probabilities[j].probability > best.probability)
	    best = i->categorical_distribution.probabilities[j];
	}
      }
      ROS_INFO_STREAM("Finished recognition.");
    }
    else
    {
      ROS_INFO_STREAM("Recognition Error!");
    }
    
    if (best.label == "unknown") 
    {
      found_objects.data = false;
      objects_pub.publish(found_objects);

      ROS_INFO_STREAM("BEST TIP: !---unknown objects---!");
    }
    else if (!found_known_object) 
    {
      found_known_object = true;

      ROS_INFO_STREAM("BEST TIP: !---" << best.label << "---!, with probability: " << best.probability);

      break;
    }
  } // for()

  if (found_known_object)
  {
    start_manipulating = true;

    ac.cancelAllGoals();
    geometry_msgs::Twist base_cmd;

    int time = nav_goal_x/0.1;
    ros::Rate loop_rate(4);

    for (int i = 0; i < time*4; i++)
    {
      ac.cancelAllGoals();
      base_cmd.linear.x = 0.1;
      base_cmd.angular.z = 0.1 * nav_goal_y;
      velocity_pub.publish(base_cmd);
      loop_rate.sleep();
    }

    base_cmd.linear.x = 0;
    velocity_pub.publish(base_cmd);

    /*move_base_msgs::MoveBaseGoal goal;

    goal.target_pose.header.frame_id = "base_footprint";
    goal.target_pose.header.stamp = ros::Time::now();

    goal.target_pose.pose.position.x = nav_goal_x;
    //goal.target_pose.pose.position.y = nav_goal_y;
    //goal.target_pose.pose.orientation.y = nav_goal_orientation;
    goal.target_pose.pose.orientation.w = 1.0;

    actionlib::SimpleClientGoalState state = actionlib::SimpleClientGoalState::ABORTED;
   
    while(state != actionlib::SimpleClientGoalState::SUCCEEDED)
    {
      ac.cancelAllGoals();
      ac.sendGoal(goal);

      ROS_INFO("Waiting for goal to accomplish...");

      ac.waitForResult();

      state = ac.getState();

      if (state == actionlib::SimpleClientGoalState::SUCCEEDED)
        ROS_INFO_STREAM("The robot moved to the object.");
      else if (state == actionlib::SimpleClientGoalState::PENDING)
        ROS_INFO_STREAM("A problem occured while moving to the object, code: PENDING");
      else if (state == actionlib::SimpleClientGoalState::ACTIVE)
        ROS_INFO_STREAM("A problem occured while moving to the object, code: ACTIVE");
      else if (state == actionlib::SimpleClientGoalState::PREEMPTED)
        ROS_INFO_STREAM("A problem occured while moving to the object, code: PREEMPTED");
      else if (state == actionlib::SimpleClientGoalState::ABORTED)
        ROS_INFO_STREAM("A problem occured while moving to the object, code: ABORTED");
      else if (state == actionlib::SimpleClientGoalState::REJECTED)
        ROS_INFO_STREAM("A problem occured while moving to the object, code: REJECTED");
      else if (state == actionlib::SimpleClientGoalState::RECALLED)
        ROS_INFO_STREAM("A problem occured while moving to the object, code: RECALLED");
      else if (state == actionlib::SimpleClientGoalState::LOST)
        ROS_INFO_STREAM("A problem occured while moving to the object, code: LOST");
      else
        ROS_INFO_STREAM("An unknown problem occured while accomplishing the goal.");
    }*/
  }


  /*pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
  pcl::visualization::CloudViewer viewer ("Cluster viewer");
  viewer.showCloud(colored_cloud);
  while (!viewer.wasStopped ())
  {
  }*/
}

int main (int argc, char** argv)
{
  ros::init (argc, argv, "object_detector");
  ros::NodeHandle n;

  ros::ServiceClient client = n.serviceClient<image_recognition_msgs::Recognize>("recognize");
  MoveBaseClient ac("move_base", true);

  while(!ac.waitForServer(ros::Duration(2.0)))
  {
    ROS_INFO("Waiting for the move_base action server to come up");
  }

  image_pub = n.advertise<sensor_msgs::Image>("roi_image", 1000);
  objects_pub = n.advertise<std_msgs::Bool>("objects", 1000);
  velocity_pub = n.advertise<geometry_msgs::Twist>("mobile_base/commands/velocity", 1000);

  //ros::Subscriber sub = n.subscribe<sensor_msgs::PointCloud2> ("/camera/depth/points", 1, boost::bind(object_detector, _1, boost::ref(client), boost::ref(n)));
  //ros::Subscriber sub = n.subscribe<sensor_msgs::Image> ("/camera/rgb/image_raw", 1, boost::bind(object_detector, _1, boost::ref(client), boost::ref(n)));
  message_filters::Subscriber<sensor_msgs::PointCloud2> depth_sub(n, "/camera/depth/points", 1);
  message_filters::Subscriber<sensor_msgs::Image> image_sub(n, "/camera/rgb/image_raw", 1);
  message_filters::Subscriber<sensor_msgs::CameraInfo> camInfo_sub(n, "/camera/rgb/camera_info", 1);

  typedef sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Image, sensor_msgs::CameraInfo> MySyncPolicy;
  // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(5), depth_sub, image_sub, camInfo_sub);

  sync.registerCallback(boost::bind(&object_detector, _1, _2, _3, boost::ref(client), boost::ref(n), boost::ref(ac)));

  ros::spin();

  return 0;
}
