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
#include <std_msgs/Bool.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_recognition_msgs/Recognize.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#define IMG_CUT 0.75
#define UNKNOWN_PROB_TRESHHOLD 0.61

using namespace message_filters;
namespace enc = sensor_msgs::image_encodings;

int first_time = 0;
ros::Time actual_time;

sensor_msgs::Image imageCb(const sensor_msgs::ImageConstPtr& msg)
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

  cv::Rect roi(100, 200, 70, 230);
  img_roi_output.header = msg->header;
  img_roi_output.encoding = enc::BGR8;
  img_roi_output.image = cv_ptr->image(roi);
  //img_roi_output.image = cv_ptr->image;
 
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
  float theta = acos((floor_plane_normal_vector[0] * xy_plane_normal_vector[0] + floor_plane_normal_vector[1] * xy_plane_normal_vector[1] + floor_plane_normal_vector[2] * xy_plane_normal_vector[2]) 
    / (sqrt(powf(floor_plane_normal_vector[0], 2) + powf(floor_plane_normal_vector[1], 2) + powf(floor_plane_normal_vector[2], 2)) * sqrt(powf(xy_plane_normal_vector[0], 2) + powf(xy_plane_normal_vector[1], 2) + powf(xy_plane_normal_vector[2], 2))));

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

// Create a region of interest from the image
// The computed region if specific for size of out robot and view angle of its sensors!  
sensor_msgs::Image select_image_area(const sensor_msgs::ImageConstPtr& img)
{
  sensor_msgs::Image::Ptr img_area = boost::make_shared<sensor_msgs::Image>();

  img_area->header = img->header;
  img_area->height = img->height * IMG_CUT;
  img_area->width = img->width - 20;
  img_area->encoding = img->encoding;
  img_area->is_bigendian = img->is_bigendian;
  img_area->step = img->step;

  img_area->data.resize(img_area->width * img_area->height * img_area->step);

  uint new_index = 0;
  for (uint row = img_area->height; row < img->height; row++)
  {
    for (uint col = 10; col < img->width-10; col++)
    {
      uint old_index = row+col*img->width;
      img_area->data[new_index++] = img->data[old_index];
    }
  }
 
  return *img_area;
}

//void object_detector(const sensor_msgs::PointCloud2ConstPtr& input, const sensor_msgs::ImageConstPtr& image, ros::ServiceClient &client, ros::NodeHandle &n)
//void object_detector(const sensor_msgs::ImageConstPtr& image, ros::ServiceClient &client, ros::NodeHandle &n)
void object_detector(const sensor_msgs::PointCloud2ConstPtr& input, ros::ServiceClient &client, ros::NodeHandle &n)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg (*input, *cloud);
  float z_after_rotation;
  static Eigen::Affine3f transform_matrix = Eigen::Affine3f::Identity();

  if (first_time == 0){
    actual_time = ros::Time::now();
    first_time = 1;
    //std::cout << "actualtime: " << actual_time << std::endl;
    transform_matrix = get_matrix(cloud, &z_after_rotation);
  }

  // Select region of interest from point cloud by filering some points
  pcl::PassThrough<pcl::PointXYZ> pass_window;
  pass_window.setInputCloud (cloud);
  pass_window.setFilterFieldName ("z");
  pass_window.setFilterLimits (0.0, 2.0);
  //pass.setFilterLimitsNegative (true);
  pass_window.filter (*cloud);

  pass_window.setInputCloud (cloud);
  pass_window.setFilterFieldName ("x");
  pass_window.setFilterLimits (-0.85, 0.85);
  //pass_window.setFilterLimitsNegative (true);
  pass_window.filter (*cloud);

  pass_window.setInputCloud (cloud);
  pass_window.setFilterFieldName ("y");
  pass_window.setFilterLimits (-0.5, 0.5);
  //pass_window.setFilterLimitsNegative (true);
  pass_window.filter (*cloud);
  

  if (ros::Time::now() - actual_time > (ros::Duration)(10)){
    actual_time = ros::Time::now();
    transform_matrix = get_matrix(cloud, &z_after_rotation);
    ROS_INFO_STREAM("Segmentation done.");
  }
 
  // Transform the point cloud
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

  pcl::IndicesPtr indices (new std::vector <int>);
  pcl::PassThrough<pcl::PointXYZ> pass2;
  pass2.setInputCloud (cloud_without_plane);
  pass2.setFilterFieldName ("z");
  pass2.setFilterLimits (0.0, 1.0);
  pass2.filter (*indices);

  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize (200);
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

  for (int i = 0; i < clusters.size(); i++)
  {
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid (*cloud_without_plane, clusters[i], centroid);
    ROS_INFO_STREAM(centroid[0] << " " <<  centroid[1] << " " <<   centroid[2] << " " <<   centroid[3]);
    ROS_INFO_STREAM("y v 2d je: " << centroid[1]/centroid[2]+240); 
  }
/*
  // Try to recognize known objects
  std::vector<image_recognition_msgs::Recognition> recognitions;
  //sensor_msgs::Image image_req = *image;
  //sensor_msgs::Image image_req = select_image_area(image);
  sensor_msgs::Image image_req = imageCb(image);
ros::Rate loop_rate(10);
ros::Publisher image_pub = n.advertise<sensor_msgs::Image>("roi_image", 1000);
image_pub.publish(image_req); 
ros::spinOnce();
loop_rate.sleep();

  //ros::Publisher objects_pub = n.advertise<std_msgs::Bool>("objects", 1000);
  std_msgs::Bool found_objects;

  ROS_INFO_STREAM("Height: " << image_req.height << ", Width: " << image_req.width << ", Step: " << image_req.step << ", Data vector size: " << image_req.data.size());
*/
  //if (clusters.size() > 0) {
    /*if (1) {

    ROS_INFO_STREAM("Object(s) in front of the robot!");

    image_recognition_msgs::Recognize srv;

    srv.request.image = image_req;
    image_recognition_msgs::CategoryProbability best;
    best.label = "unknown";
    best.probability = UNKNOWN_PROB_TRESHHOLD;
 
    if (client.call(srv)) {
      recognitions = srv.response.recognitions;

      for(std::vector<image_recognition_msgs::Recognition>::iterator i = recognitions.begin(); i != recognitions.end(); ++i) {
	best.label = "unknown";
	//best.probability = i->categorical_distribution.unknown_probability;
        best.probability = UNKNOWN_PROB_TRESHHOLD;

	for (unsigned int j = 0; j < i->categorical_distribution.probabilities.size(); j++) {
	  if (i->categorical_distribution.probabilities[j].probability > best.probability)
	    best = i->categorical_distribution.probabilities[j];
	}
      }
    }

    if (best.label == "unknown") {

      found_objects.data = false;

      ROS_INFO_STREAM("Best tip: --unknown objects--");
    }
    else {
    
      found_objects.data = true;

      ROS_INFO_STREAM("Best tip: --" << best.label << "--, with probability: " << best.probability);
    }
  } 
  else {
    ROS_INFO_STREAM("Free space in front of the robot.");
  }

  objects_pub.publish(found_objects);
  ros::spinOnce();
*/
  /*pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
  pcl::visualization::CloudViewer viewer ("Cluster viewer");
  viewer.showCloud(colored_cloud);
  while (!viewer.wasStopped ())
  {
  }*/
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "object_detector");
  ros::NodeHandle n;
  ros::ServiceClient client = n.serviceClient<image_recognition_msgs::Recognize>("recognize");

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = n.subscribe<sensor_msgs::PointCloud2> ("/camera/depth/points", 1, boost::bind(object_detector, _1, boost::ref(client), boost::ref(n)));
  //ros::Subscriber sub = n.subscribe<sensor_msgs::Image> ("/camera/rgb/image_raw", 1, boost::bind(object_detector, _1, boost::ref(client), boost::ref(n)));
  /*message_filters::Subscriber<sensor_msgs::PointCloud2> depth_sub(n, "/camera/depth/points", 1);
  message_filters::Subscriber<sensor_msgs::Image> image_sub(n, "/camera/rgb/image_raw", 1);

  typedef sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Image> MySyncPolicy;
  // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), depth_sub, image_sub);

  sync.registerCallback(boost::bind(&object_detector, _1, _2, boost::ref(client), boost::ref(n)));
*/
  // Spin
  ros::spin();

  return 0;
}
