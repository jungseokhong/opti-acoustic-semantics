#!/usr/bin/env python  
import rospy

import math
import tf2_ros
import tf.transformations as tfm
import geometry_msgs.msg


def transform_to_matrix(transform):
    translation = [
        transform.transform.translation.x,
        transform.transform.translation.y,
        transform.transform.translation.z
    ]
    rotation = [
        transform.transform.rotation.x,
        transform.transform.rotation.y,
        transform.transform.rotation.z,
        transform.transform.rotation.w
    ]
    translation_matrix = tfm.translation_matrix(translation)
    rotation_matrix = tfm.quaternion_matrix(rotation)
    return tfm.concatenate_matrices(translation_matrix, rotation_matrix)

def matrix_to_transform(matrix):
    translation = tfm.translation_from_matrix(matrix)
    rotation = tfm.quaternion_from_matrix(matrix)
    transform = geometry_msgs.msg.Transform()
    transform.translation.x = translation[0]
    transform.translation.y = translation[1]
    transform.translation.z = translation[2]
    transform.rotation.x = rotation[0]
    transform.rotation.y = rotation[1]
    transform.rotation.z = rotation[2]
    transform.rotation.w = rotation[3]
    return transform

if __name__ == '__main__':
    rospy.init_node('tf2_listener')

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    tf_broadcaster = tf2_ros.TransformBroadcaster()


    rate = rospy.Rate(240.0)
    while not rospy.is_shutdown():
        try:
            # trans = tfBuffer.lookup_transform('world', 'ZED2i', rospy.Time())
            # print(trans)

            # Get transform from /world to /ZED2i
            common_time = rospy.Time()
            trans_world_to_zed = tfBuffer.lookup_transform('world', 'ZED2i', common_time)
            # print(trans_world_to_zed)

            # Get transform from /odom to /base_link
            # trans_odom_to_base = tfBuffer.lookup_transform('odom', 'base_link', common_time)
            # print(trans_odom_to_base)

            # Get transform from /ZED2i to /zed2i_left_camera_optical_frame
            trans_zed_to_optical = tfBuffer.lookup_transform('ZED2i', 'zed2i_left_camera_optical_frame', common_time)
            
            # Get transform from /zed2i_left_camera_optical_frame to /zed2i_base_link
            # trans_optical_to_base = tfBuffer.lookup_transform('zed2i_left_camera_optical_frame', 'zed2i_base_link', common_time)
            
            # trans_optical_to_base.transform.translation.x = 0.060
            # trans_optical_to_base.transform.translation.y = 0.015
            # trans_optical_to_base.transform.translation.z = 0.010
            # trans_optical_to_base.transform.rotation.x = 0.500
            # trans_optical_to_base.transform.rotation.y = -0.500
            # trans_optical_to_base.transform.rotation.z = 0.500
            # trans_optical_to_base.transform.rotation.w = 0.500

            translation_optical_to_base = [0.060, 0.015, 0.010]
            rotation_optical_to_base = [0.500, -0.500, 0.500, 0.500]
            translation_matrix_optical_to_base = tfm.translation_matrix(translation_optical_to_base)
            rotation_matrix_optical_to_base = tfm.quaternion_matrix(rotation_optical_to_base)

            
            # Get transform from /base_link to /zed2i_base_link
            ## this is Identity transform
            # trans_base_to_zed_base = tfBuffer.lookup_transform('base_link', 'zed2i_base_link', common_time)

            # Convert transforms to matrices
            matrix_world_to_zed = transform_to_matrix(trans_world_to_zed)
            matrix_zed_to_optical = transform_to_matrix(trans_zed_to_optical)
            matrix_optical_to_base = tfm.concatenate_matrices(translation_matrix_optical_to_base, rotation_matrix_optical_to_base)
            # matrix_base_to_zed_base = transform_to_matrix(trans_base_to_zed_base)
            # matrix_odom_to_base = transform_to_matrix(trans_odom_to_base)

            # Combine the transformations
            matrix_world_to_base = tfm.concatenate_matrices(
                matrix_world_to_zed,
                matrix_zed_to_optical,
                matrix_optical_to_base,
            )

            # Convert the resulting matrix to a Transform message
            combined_transform = matrix_to_transform(matrix_world_to_base)

            # Create a TransformStamped message for broadcasting
            transform_stamped = geometry_msgs.msg.TransformStamped()
            transform_stamped.header.stamp = rospy.Time.now()
            transform_stamped.header.frame_id = 'world'
            transform_stamped.child_frame_id = 'base_link'
            transform_stamped.transform = combined_transform

            # Broadcast the transform
            tf_broadcaster.sendTransform(transform_stamped)


        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("TF lookup failed: {}".format(e))
            rate.sleep()
            continue

        
        # msg = geometry_msgs.msg.Twist()

        # msg.angular.z = 4 * math.atan2(trans.transform.translation.y, trans.transform.translation.x)
        # msg.linear.x = 0.5 * math.sqrt(trans.transform.translation.x ** 2 + trans.transform.translation.y ** 2)

        # turtle_vel.publish(msg)

        rate.sleep()