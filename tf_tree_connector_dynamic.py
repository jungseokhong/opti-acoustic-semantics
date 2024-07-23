#!/usr/bin/env python

import rospy
import tf2_ros
import tf
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import message_filters
import tf_conversions

def callback(odom_msg, world_msg):

    tfBuffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tfBuffer)
    # tf_listener = tf2_ros.TransformListener()
    tf_broadcaster = tf2_ros.TransformBroadcaster()

    # Get transform from /world to /ZED2i
    (trans_world_to_zed, rot_world_to_zed) = tfBuffer.waitForTransform('world', 'ZED2i', rospy.Time())

    # Get transform from /odom to /base_link
    (trans_odom_to_base, rot_odom_to_base) = tfBuffer.waitForTransform('odom', 'base_link', rospy.Time())
    
    # Get transform from /ZED2i to /zed2i_left_camera_optical_frame
    (trans_zed_to_optical, rot_zed_to_optical) = tfBuffer.waitForTransform('ZED2i', 'zed2i_left_camera_optical_frame', rospy.Time())
    
    # Get transform from /zed2i_left_camera_optical_frame to /zed2i_base_link
    (trans_optical_to_base, rot_optical_to_base) = tfBuffer.waitForTransform('zed2i_left_camera_optical_frame', 'zed2i_base_link', rospy.Time())
    
    # Get transform from /base_link to /zed2i_base_link
    (trans_base_to_zed_base, rot_base_to_zed_base) = tfBuffer.waitForTransform('base_link', 'zed2i_base_link', rospy.Time())

    # Combine transforms to get /world to /odom
    # You may need to adjust the combination order based on your specific transform chain
    world_to_odom_trans = tf.transformations.concatenate_matrices(
        tf.transformations.translation_matrix(trans_world_to_zed),
        tf.transformations.quaternion_matrix(rot_world_to_zed),
        tf.transformations.translation_matrix(trans_zed_to_optical),
        tf.transformations.quaternion_matrix(rot_zed_to_optical),
        tf.transformations.translation_matrix(trans_optical_to_base),
        tf.transformations.quaternion_matrix(rot_optical_to_base),
        tf.transformations.translation_matrix(trans_base_to_zed_base),
        tf.transformations.quaternion_matrix(rot_base_to_zed_base),
        tf.transformations.translation_matrix(trans_odom_to_base),
        tf.transformations.quaternion_matrix(rot_odom_to_base)
    )


    # Extract the combined translation and rotation
    combined_translation = tf.transformations.translation_from_matrix(world_to_odom_trans)
    combined_rotation = tf.transformations.quaternion_from_matrix(world_to_odom_trans)
    print(combined_translation, combined_rotation)

    # Publish the transform
    transform = TransformStamped()
    transform.header.stamp = odom_msg.header.stamp  # Use synchronized timestamp
    transform.header.frame_id = 'world'
    transform.child_frame_id = 'odom'
    transform.transform.translation.x = combined_translation[0]
    transform.transform.translation.y = combined_translation[1]
    transform.transform.translation.z = combined_translation[2]
    transform.transform.rotation.x = combined_rotation[0]
    transform.transform.rotation.y = combined_rotation[1]
    transform.transform.rotation.z = combined_rotation[2]
    transform.transform.rotation.w = combined_rotation[3]

    tf_broadcaster.sendTransform(transform)

def main():
    rospy.init_node('dynamic_tf_broadcaster')

    odom_sub = message_filters.Subscriber('/zed2i/zed_node/odom', Odometry)
    world_sub = message_filters.Subscriber('/ZED2i/world', PoseStamped)

    ts = message_filters.ApproximateTimeSynchronizer([odom_sub, world_sub], queue_size=10, slop=0.1)
    ts.registerCallback(callback)

    rospy.spin()

if __name__ == '__main__':
    main()
