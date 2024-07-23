#!/usr/bin/env python  
import rospy

import math
import tf2_ros
import tf.transformations as tfm
import geometry_msgs.msg
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import message_filters


class Transformer:
    def __init__(self):
        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)
        # tf_listener = tf2_ros.TransformListener()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.odom_sub = message_filters.Subscriber('/zed2i/zed_node/odom', Odometry)
        self.world_sub = message_filters.Subscriber('/ZED2i/world', PoseStamped)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            (self.odom_sub, self.world_sub), 500, 0.005
        )  # 0.025 need to reduce this time difference
        # # need to update so it can handle time offset/pass time offset

        self.sync.registerCallback(self.forward_pass)


    def transform_to_matrix(self, transform):
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

    def matrix_to_transform(self, matrix):
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

    def forward_pass(self, odom_msg, world_msg):
        # Get transform from /world to /ZED2i

        try:
            print("Received messages")
            common_time = odom_msg.header.stamp
            trans_world_to_zed = self.tfBuffer.lookup_transform('world', 'ZED2i', common_time)
            # print(trans_world_to_zed)

            # Get transform from /odom to /base_link
            trans_odom_to_base = self.tfBuffer.lookup_transform('odom', 'base_link', common_time)
            # print(trans_odom_to_base)

            # Get transform from /ZED2i to /zed2i_left_camera_optical_frame
            trans_zed_to_optical = self.tfBuffer.lookup_transform('ZED2i', 'zed2i_left_camera_optical_frame', common_time)
            translation_optical_to_base = [0.060, 0.015, 0.010]
            rotation_optical_to_base = [0.500, -0.500, 0.500, 0.500]
            translation_matrix_optical_to_base = tfm.translation_matrix(translation_optical_to_base)
            rotation_matrix_optical_to_base = tfm.quaternion_matrix(rotation_optical_to_base)

            
            # Get transform from /base_link to /zed2i_base_link
            trans_base_to_zed_base = self.tfBuffer.lookup_transform('base_link', 'zed2i_base_link', common_time)

            # Convert transforms to matrices
            matrix_world_to_zed = self.transform_to_matrix(trans_world_to_zed)
            matrix_zed_to_optical = self.transform_to_matrix(trans_zed_to_optical)
            matrix_optical_to_base = tfm.concatenate_matrices(translation_matrix_optical_to_base, rotation_matrix_optical_to_base)
            matrix_base_to_zed_base = self.transform_to_matrix(trans_base_to_zed_base)
            matrix_odom_to_base = self.transform_to_matrix(trans_odom_to_base)

            # Combine the transformations
            matrix_world_to_odom = tfm.concatenate_matrices(
                matrix_world_to_zed,
                matrix_zed_to_optical,
                matrix_optical_to_base,
                matrix_base_to_zed_base,
                matrix_odom_to_base
            )

            # Convert the resulting matrix to a Transform message
            combined_transform = self.matrix_to_transform(matrix_world_to_odom)

            # Create a TransformStamped message for broadcasting
            transform_stamped = geometry_msgs.msg.TransformStamped()
            transform_stamped.header.stamp = rospy.Time.now()
            transform_stamped.header.frame_id = 'world'
            transform_stamped.child_frame_id = 'odom'
            transform_stamped.transform = combined_transform

            # Broadcast the transform
            self.tf_broadcaster.sendTransform(transform_stamped)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("TF lookup failed: {}".format(e))


if __name__ == '__main__':
    rospy.init_node('tf2_listener')
    transformer = Transformer()
    while not rospy.is_shutdown():
        rospy.spin()
        rospy.sleep(0.1)