#!/usr/bin/env python3

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
import tf_conversions

def publish_static_transform():
    rospy.init_node('static_tf_broadcaster')

    broadcaster = tf2_ros.StaticTransformBroadcaster()
    static_transform_stamped = TransformStamped()

    static_transform_stamped.header.stamp = rospy.Time.now()
    static_transform_stamped.header.frame_id = "zed2i_left_camera_optical_frame"
    static_transform_stamped.child_frame_id = "zed2i_base_link"

    # Set your known transform here
    static_transform_stamped.transform.translation.x = 0.060
    static_transform_stamped.transform.translation.y = 0.015
    static_transform_stamped.transform.translation.z = 0.010

    static_transform_stamped.transform.rotation.x = 0.500
    static_transform_stamped.transform.rotation.y = -0.500
    static_transform_stamped.transform.rotation.z = 0.500
    static_transform_stamped.transform.rotation.w = 0.500

    broadcaster.sendTransform(static_transform_stamped)

    rospy.spin()

if __name__ == '__main__':
    publish_static_transform()
