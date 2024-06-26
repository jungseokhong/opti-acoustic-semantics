#!/usr/bin/env python3

import message_filters
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
import cv2
from semanticslam_ros.msg import MapInfo
from sensor_msgs.msg import CameraInfo
from PIL import Image as PILImage
from sensor_msgs.msg import Image as RosImage
import sys
from scipy.spatial.transform import Rotation as R

# Define camera matrix K
# fx = 260.9886474609375
# fy = 260.9886474609375
# cx = 322.07867431640625
# cy = 179.7025146484375
# K = np.array([[fx, 0, cx],
#               [0, fy, cy],
#               [0, 0, 1]])

IMG_HEIGHT = 540# 360
IMG_WIDTH = 960 #640

# D: [0.0, 0.0, 0.0, 0.0, 0.0]
# K: [527.150146484375, 0.0, 485.47442626953125, 0.0, 527.150146484375, 271.170166015625, 0.0, 0.0, 1.0]
# R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
# P: [527.150146484375, 0.0, 485.47442626953125, 0.0, 0.0, 527.150146484375, 271.170166015625, 0.0, 0.0, 0.0, 1.0, 0.0]
# [TODO] should subscribe to the camera info topic to get the camera matrix K rather than hardcoding it
# height: 540
# width: 960

class Compare2DMapAndImage:
    def __init__(self):
        rospy.loginfo("compare_map_img service started")
        self.K = np.zeros((3, 3))
        fx = 527.150146484375
        fy = 527.150146484375
        cx = 485.47442626953125
        cy = 271.170166015625

        # fx = 260.9886474609375
        # fy = 260.9886474609375
        # cx = 322.07867431640625
        # cy = 179.7025146484375

        self.K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]])

        self.img_height = 540
        self.img_width = 960
        self.bridge = CvBridge()
        self.yoloimg_sub = message_filters.Subscriber('/camera/yolo_img', RosImage)
        self.mapinfo_sub = message_filters.Subscriber('/mapinfo', MapInfo)
        # self.cam_info_sub = message_filters.Subscriber('/zed2i/zed_node/rgb/camera_info', CameraInfo, queue_size=1)
        self.compare_pub = rospy.Publisher("/compareresults", RosImage, queue_size=10)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            (self.yoloimg_sub, self.mapinfo_sub), 100, 0.1
        ) #0.025 need to reduce this time difference
        # # need to update so it can handle time offset/pass time offset

        # self.sync = message_filters.ApproximateTimeSynchronizer(
        #     (self.yoloimg_sub, self.mapinfo_sub, self.cam_info_sub), 100, 0.1
        # ) 


        self.sync.registerCallback(self.forward_pass)

    # def forward_pass(self, yoloimg : RosImage, map_info : MapInfo, cam_info: CameraInfo) -> None:
    #     # Convert ROS Image to OpenCV Image
    #     yoloimg_cv = self.bridge.imgmsg_to_cv2(yoloimg, desired_encoding='bgr8')
    #     # print(map_info)

    #     # Extract data from map_info
    #     position, orientation, landmark_points, landmark_classes = self.parse_data(map_info)
    #     # Project landmarks to the image
    #     projected_image = self.projectLandmarksToImage(position, orientation, landmark_points, landmark_classes, img = yoloimg_cv)
    #     # projected_image = self.projectLandmarksToImage(position, orientation, landmark_points, landmark_classes)


    #     # Combine yoloimg_cv and projected_image side by side
    #     # if we want to display the images side by side
    #     combined_image = self.combine_images(yoloimg_cv, projected_image)
    #     fx = cam_info.K[0]
    #     fy = cam_info.K[4]
    #     cx = cam_info.K[2]
    #     cy = cam_info.K[5]
    #     self.img_height = cam_info.height
    #     self.img_width = cam_info.width
    #     self.K = np.array([[fx, 0, cx],
    #                     [0, fy, cy],
    #                     [0, 0, 1]])

    #     # Convert the combined OpenCV image back to ROS Image
    #     try:
    #         ros_image = self.bridge.cv2_to_imgmsg(combined_image, "bgr8")
    #     except self.bridge.CvBridgeError as e:
    #         rospy.logerr("Failed to convert combined image to ROS Image: %s", e)
    #         return

    #     # Publish the ROS Image
    #     self.compare_pub.publish(ros_image)

    def forward_pass(self, yoloimg : RosImage, map_info : MapInfo) -> None:
        # Convert ROS Image to OpenCV Image
        yoloimg_cv = self.bridge.imgmsg_to_cv2(yoloimg, desired_encoding='bgr8')
        # print(map_info)

        # Extract data from map_info
        position, orientation, landmark_points, landmark_classes = self.parse_data(map_info)
        # Project landmarks to the image
        projected_image = self.projectLandmarksToImage(position, orientation, landmark_points, landmark_classes, img = yoloimg_cv)
        # projected_image = self.projectLandmarksToImage(position, orientation, landmark_points, landmark_classes)


        # Combine yoloimg_cv and projected_image side by side
        # if we want to display the images side by side
        # combined_image = self.combine_images(yoloimg_cv, projected_image)
        combined_image = projected_image

        # Convert the combined OpenCV image back to ROS Image
        try:
            ros_image = self.bridge.cv2_to_imgmsg(combined_image, "bgr8")
        except self.bridge.CvBridgeError as e:
            rospy.logerr("Failed to convert combined image to ROS Image: %s", e)
            return

        # Publish the ROS Image
        self.compare_pub.publish(ros_image)


    def parse_data(self, map_info):

        landmark_points = []
        landmark_classes = []
        position_data = []
        orientation_data = []

        num_landmarks = len(map_info.landmark_points)
        for i in range(num_landmarks):
            landmark_points.append(map_info.landmark_points[i].x)
            landmark_points.append(map_info.landmark_points[i].y)
            landmark_points.append(map_info.landmark_points[i].z)
            landmark_classes.append(map_info.landmark_classes[i])
        
        orientation_data.append(map_info.pose.orientation.x)
        orientation_data.append(map_info.pose.orientation.y)
        orientation_data.append(map_info.pose.orientation.z)
        orientation_data.append(map_info.pose.orientation.w)

        position_data.append(map_info.pose.position.x)
        position_data.append(map_info.pose.position.y)
        position_data.append(map_info.pose.position.z)

        # Convert lists to numpy arrays
        landmark_points_array = np.array(landmark_points).reshape(-1, 3)
        landmark_classes_array = np.array(landmark_classes)
        position_array = np.array(position_data)
        orientation_array = np.array(orientation_data)
        # Reorder the orientation array
        # orientation_array = np.roll(orientation_array, -3) # for w,x,y,z convention

        return position_array, orientation_array, landmark_points_array, landmark_classes_array

    def projectLandmarksToImage(self, position, orientation, landmark_points, landmark_classes, img=None):

        # Quaternion to rotation matrix conversion
        q = orientation

        r = R.from_quat(q)
        rotation_matrix = r.as_matrix() # body to world

        camToBody = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]) # R_B_C
        print(f' Cam to Body: {camToBody}')
        new_rotation_matrix = np.matmul(rotation_matrix, camToBody)

        R_W_B = rotation_matrix
        R_B_W = np.linalg.inv(R_W_B)
        R_B_C = camToBody
        R_C_B = np.linalg.inv(R_B_C)
        R_C_W = np.matmul(R_C_B, R_B_W)

        # Convert rotation matrix to Rodrigues vector
        R_vec, _ = cv2.Rodrigues(new_rotation_matrix)
        t_vec = position

        # Project points
        dist_coeffs = np.zeros(4)  # Assuming no lens distortion

        # building projection matrix
        RT = np.zeros([3,4])
        RT[:3, :3] = R_C_W # np.linalg.inv(new_rotation_matrix)
        RT[:3, 3] = -R_C_W@t_vec
        print(f'RT: {RT}')

        # Step 1: Transpose the matrix to make it 3xN
        transposed_points = landmark_points.T
        # Step 2: Add a row of ones to make it 4xN
        homogeneous_points = np.vstack([transposed_points, np.ones(transposed_points.shape[1])])

        # points_2d, _ = cv2.projectPoints(landmark_points, new_rotation_matrix, t_vec, K, dist_coeffs)
        points_2d_homo = self.K @ RT @ homogeneous_points
        print(points_2d_homo.shape, points_2d_homo.T)

        # height = 540# 360
        # width = 960 #640

        # Initialize a blank image
        if img is not None and img.size > 0:
            projected_image = img
        else:

            projected_image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

        # Draw each point and class label on the image
        for i, point in enumerate(points_2d_homo.T):
            x, y = int(point[0]/point[2]), int(point[1]/point[2])
            if 0 <= x < self.img_width and 0 <= y < self.img_height:
                cv2.circle(projected_image, (x, y), 4, (0, 255, 0), -1)  # Green dot
                cv2.putText(projected_image, landmark_classes[i], (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


        return projected_image

    def combine_images(self, img1, img2):
        # Ensure both images are the same height
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape

        # Resize images if they do not match in dimensions
        if h1 != h2:
            img1 = cv2.resize(img1, (w1, h2), interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, (w2, h2), interpolation=cv2.INTER_LINEAR)

        # Concatenate images horizontally
        combined_image = np.hstack((img1, img2))
        return combined_image


if __name__ == "__main__":
    rospy.init_node("information_comparison")
    detector = Compare2DMapAndImage()
    rospy.spin()