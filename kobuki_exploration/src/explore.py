#!/usr/bin/env python

import roslib; roslib.load_manifest('kobuki_exploration')
import rospy
import time
import tf
import actionlib
import frontier_exploration.msg
from std_msgs.msg import Bool
from geometry_msgs.msg import PolygonStamped, PointStamped, Polygon, Point32
    
class PR2RobotBrain:

    def __init__(self, boundary, center):
    
        self.tfl = tf.TransformListener()
        
        self.boundary = boundary
        self.center = center
        self.action_client = actionlib.SimpleActionClient('explore_server', frontier_exploration.msg.ExploreTaskAction)
        
        rospy.Subscriber("objects", Bool, self.foundObjects)
        
        self.stop_exploring = False
        
    def getReady(self):
    
        self.action_client.wait_for_server()
    
    
    def explorationFeedbackCallback(self, fb):
  
        pass
    
    def foundObjects(self, data):
    
        if data.data == True:
            rospy.loginfo("Setting stop_exploration to True.")
            self.stop_exploring = True
        else:
            rospy.loginfo("Setting stop_exploration to False.")
            self.stop_exploring = False
    
    def loop(self):
    
        self.getReady()
    
        while not rospy.is_shutdown():
        
            # Explore until object is found
            if self.stop_exploring == False:
                self.explore()
            
            # Object found - start manipulation / call Toad
            # TBD
        
        
    def explore(self):
    
        while not (self.stop_exploring or rospy.is_shutdown()):
    
            goal = frontier_exploration.msg.ExploreTaskGoal()
            goal.explore_boundary = self.boundary
            goal.explore_center = self.center
            
            if not self.stop_exploring:
                self.action_client.send_goal(goal, feedback_cb=self.explorationFeedbackCallback)
            rospy.loginfo("Exploration started.") 
            
            while not self.action_client.wait_for_result(rospy.Duration(0.1)):

                if rospy.is_shutdown():
                
                    self.action_client.cancel_all_goals()
                    break

                elif self.stop_exploring:

                    break
            
            rospy.loginfo("Exploration finished.") 
        
    
def main():
  
    rospy.init_node('kobuki_exploration')
    
    area_to_explore = PolygonStamped()
    center_point = PointStamped()
    
    now = rospy.Time.now()
    
    area_to_explore.header.stamp = now
    area_to_explore.header.frame_id = "map"
    points = [Point32(-2.65, -2.56, 0.0),
              Point32(5.41, -2.7, 0.0),
              Point32(5.57, 4.44, 0.0),
              Point32(-2.75, 4.37, 0.0)]
              
    area_to_explore.polygon = Polygon(points)
    
    center_point.header = area_to_explore.header
    center_point.point.x = 1.83
    center_point.point.y = 1.57
    center_point.point.z = 0.0
    
    brain = PR2RobotBrain(area_to_explore, center_point)
    brain.getReady()
    brain.loop()
    
if __name__ == '__main__':
    main()
