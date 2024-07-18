#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import TransformStamped, PoseStamped

def transform_callback(transform_msg):
    pose_msg = PoseStamped()
    pose_msg.header = transform_msg.header
    pose_msg.header.frame_id = "odom"  # Set the frame_id to "odom"
    pose_msg.pose.position.x = transform_msg.transform.translation.x
    pose_msg.pose.position.y = transform_msg.transform.translation.y
    pose_msg.pose.position.z = transform_msg.transform.translation.z
    pose_msg.pose.orientation = transform_msg.transform.rotation

    pose_pub.publish(pose_msg)

if __name__ == '__main__':
    rospy.init_node('transform_to_pose')

    pose_pub = rospy.Publisher('/vicon/pose_stamped', PoseStamped, queue_size=10)
    rospy.Subscriber('/vicon/ZED_camera/ZED_camera', TransformStamped, transform_callback)

    rospy.spin()
