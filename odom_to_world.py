#!/usr/bin/env python3

import rospy
import tf
import yaml

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def apply_transformation(listener, broadcaster, transformation, parent_frame, child_frame):
    rate = rospy.Rate(10.0)
    
    while not rospy.is_shutdown():
        try:
            # Get the current transform between 'odom' and 'world'
            (trans, rot) = listener.lookupTransform('world', 'odom', rospy.Time(0))
            
            # Combine the current transform with the new transformation
            combined_trans = (
                trans[0] + transformation['x'],
                trans[1] + transformation['y'],
                trans[2] + transformation['z']
            )
            combined_rot = tf.transformations.quaternion_multiply(rot, [
                transformation['qx'], transformation['qy'], transformation['qz'], transformation['qw']
            ])
            
            # Broadcast the combined transform
            broadcaster.sendTransform(combined_trans,
                                      combined_rot,
                                      rospy.Time.now(),
                                      child_frame,
                                      parent_frame)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
        
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('transform_broadcaster')
    
    file_path = '/home/jungseok/.ros/easy_handeye/eye_in_hand_eye_on_hand.yaml'
    yaml_data = load_yaml(file_path)
    
    transformation = yaml_data['transformation']
    parent_frame = 'world'
    child_frame = 'odom_transformed'

    listener = tf.TransformListener()
    broadcaster = tf.TransformBroadcaster()

    try:
        apply_transformation(listener, broadcaster, transformation, parent_frame, child_frame)
    except rospy.ROSInterruptException:
        pass
