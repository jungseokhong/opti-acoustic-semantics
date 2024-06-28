#!/usr/bin/env python3
import os
from pathlib import Path

import message_filters
import numpy as np
import rospy
from cv_bridge import CvBridge
import cv2
from openai import OpenAI
from semanticslam_ros.msg import MapInfo, ObjectsVector

from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as RosImage
from scipy.spatial.transform import Rotation as R
from semanticslam_ros.srv import RemoveClass, RemoveClassRequest
from ast import literal_eval

from vlm_filter_utils import vision_filter

import sys
# sys.path.append("/home/beantown/ran/llm-mapping")
sys.path.append("/home/jungseok/git/llm-mapping")
from beantown_agent.map_agent import vision_agent
from beantown_agent.agent_utils import return_str

OPENAI_API_BASE = "https://api.openai.com/v1"
os.environ['OPENAI_API_BASE'] = OPENAI_API_BASE

# K: [527.150146484375, 0.0, 485.47442626953125, 0.0, 527.150146484375, 271.170166015625, 0.0, 0.0, 1.0]
# [TODO] should subscribe to the camera info topic to get the camera matrix K rather than hardcoding it


def generate_unique_colors(num_colors):
    hsv_colors = [(i * 180 // num_colors, 255, 255) for i in
                  range(num_colors)]  # Generate colors with maximum saturation and value
    bgr_colors = [cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0][0] for hsv in hsv_colors]
    return [tuple(color.tolist()) for color in bgr_colors]  # Convert each color from numpy array to tuple


class Compare2DMapAndImage:
    def __init__(self):

        self.save_projections = True
        #self.output_dir = Path("/home/beantown/datasets/llm_data/rosbag_output/")
        self.output_dir = Path("/home/jungseok/data/llm_data/rosbag_output/")
        rospy.loginfo("compare_map_img service started")
        self.K = np.zeros((3, 3))
        fx = 527.150146484375
        fy = 527.150146484375
        cx = 485.47442626953125
        cy = 271.170166015625

        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])

        self.img_height = 540
        self.img_width = 960
        self.max_time_diff = 0
        self.min_time_diff = 10
        self.classlist = []
        self.classstring = ""

        self.num_classes = 20  # Example: Specify the number of classes
        self.colors = generate_unique_colors(self.num_classes)  # Generate unique colors for these classes

        self.bridge = CvBridge()
        # self.yoloimg_sub = message_filters.Subscriber('/camera/yolo_img', RosImage)
        self.yoloimg_sub = message_filters.Subscriber('/zed2i/zed_node/rgb/image_rect_color', RosImage)
        self.mapinfo_sub = message_filters.Subscriber('/mapinfo', MapInfo)
        self.classlist_sub = message_filters.Subscriber('/camera/objects', ObjectsVector)

        # Separate subscriber for CameraInfo
        self.cam_info_sub = rospy.Subscriber('/zed2i/zed_node/rgb/camera_info', CameraInfo, self.camera_info_callback)

        self.compare_pub = rospy.Publisher("/compareresults", RosImage, queue_size=10)

        # Use a cache for the mapinfo_sub to store messages
        self.mapinfo_cache = message_filters.Cache(self.mapinfo_sub, 500)
        self.yoloimg_cache = message_filters.Cache(self.yoloimg_sub, 500)
        self.classlist_cache = message_filters.Cache(self.classlist_sub, 500)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            (self.yoloimg_sub, self.mapinfo_cache, self.classlist_cache), 500, 0.005
        )  # 0.025 need to reduce this time difference
        # # need to update so it can handle time offset/pass time offset

        self.sync.registerCallback(self.forward_pass)

        self.frame_num = 0
        self.vlm_input = None

        self.client = OpenAI()
        self.vlm_filter = vision_agent(vision_filter)

    def camera_info_callback(self, cam_info: CameraInfo):
        # Update camera intrinsic matrix K
        fx = cam_info.K[0]
        fy = cam_info.K[4]
        cx = cam_info.K[2]
        cy = cam_info.K[5]
        self.img_height = cam_info.height
        self.img_width = cam_info.width
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])

    def forward_pass(self, yoloimg: RosImage, map_info: MapInfo, objects_info: ObjectsVector) -> None:

        # Print the timestamps of the messages
        yoloimg_time = yoloimg.header.stamp
        mapinfo_time = map_info.header.stamp
        # rospy.loginfo("YOLO image timestamp: %s", yoloimg_time)
        # rospy.loginfo("Map info timestamp: %s", mapinfo_time)

        # Calculate and print the time difference
        time_diff = (yoloimg_time - mapinfo_time).to_sec()
        # rospy.loginfo("Time difference between YOLO image and Map info: %.6f seconds", time_diff)
        # rospy.loginfo("Max and min time difference: %.6f, %.6f", self.max_time_diff, self.min_time_diff)
        if abs(time_diff) > self.max_time_diff:
            self.max_time_diff = abs(time_diff)
            # rospy.loginfo("New max time difference: %.6f seconds", self.max_time_diff)
        elif abs(time_diff) < self.min_time_diff:
            self.min_time_diff = abs(time_diff)
            # rospy.loginfo("New min time difference: %.6f seconds", self.min_time_diff)

        # Convert ROS Image to OpenCV Image
        yoloimg_cv = self.bridge.imgmsg_to_cv2(yoloimg, desired_encoding='bgr8')
        # print(map_info)

        # Extract data from map_info
        position, orientation, landmark_points, landmark_classes, landmark_widths, landmark_heights = self.parse_data(
            map_info)
        # Project landmarks to the image
        projected_image = self.projectLandmarksToImage(position, orientation, landmark_points, landmark_classes,
                                                       landmark_widths, landmark_heights, img=yoloimg_cv)



        # Combine yoloimg_cv and projected_image side by side
        # if we want to display the images side by side
        # combined_image = self.combine_images(yoloimg_cv, projected_image)
        combined_image = projected_image
        self.classstring = objects_info.classlist.data

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
        landmark_widths = []
        landmark_heights = []

        num_landmarks = len(map_info.landmark_points)
        for i in range(num_landmarks):
            landmark_points.append(map_info.landmark_points[i].x)
            landmark_points.append(map_info.landmark_points[i].y)
            landmark_points.append(map_info.landmark_points[i].z)
            landmark_classes.append(map_info.landmark_classes[i])
            landmark_widths.append(map_info.landmark_widths[i])
            landmark_heights.append(map_info.landmark_heights[i])

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
        landmark_widths_array = np.array(landmark_widths)
        landmark_heights_array = np.array(landmark_heights)
        # Reorder the orientation array
        # orientation_array = np.roll(orientation_array, -3) # for w,x,y,z convention

        return position_array, orientation_array, landmark_points_array, landmark_classes_array, landmark_widths_array, landmark_heights_array

    def projectLandmarksToImage(self, position, orientation, landmark_points, landmark_classes, landmark_widths,
                                landmark_heights, img=None):

        # Quaternion to rotation matrix conversion
        q = orientation

        r = R.from_quat(q)
        rotation_matrix = r.as_matrix()  # body to world

        camToBody = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])  # R_B_C
        # print(f' Cam to Body: {camToBody}')
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
        RT = np.zeros([3, 4])
        RT[:3, :3] = R_C_W  # np.linalg.inv(new_rotation_matrix)
        RT[:3, 3] = -R_C_W @ t_vec
        # print(f'RT: {RT}')

        # Step 1: Transpose the matrix to make it 3xN
        transposed_points = landmark_points.T
        # Step 2: Add a row of ones to make it 4xN
        homogeneous_points = np.vstack([transposed_points, np.ones(transposed_points.shape[1])])

        # points_2d, _ = cv2.projectPoints(landmark_points, new_rotation_matrix, t_vec, K, dist_coeffs)
        points_2d_homo = self.K @ RT @ homogeneous_points
        # print(points_2d_homo.shape, points_2d_homo.T)


        # Initialize a blank image
        if img is not None and img.size > 0:
            projected_image = img.copy()
        else:
            projected_image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        # Map class indices to colors
        class_to_color = {i: self.colors[i % len(self.colors)] for i in range(len(landmark_classes))}
        # print(f"landmark widths {landmark_widths} landmark_heights {landmark_heights}, 2dpoints {points_2d_homo}")

        # Iterate over each projected point
        json_out = {}
        obj = []
        for i, (point_3d, point_2d) in enumerate(zip(landmark_points, points_2d_homo.T)):
            x, y = int(point_2d[0] / point_2d[2]), int(point_2d[1] / point_2d[2])
            # Compute the Euclidean distance between the point and the camera position
            Z = np.linalg.norm(point_3d - position)

            # Scale widths and heights based on the depth
            scale_factor_w = self.K[
                                 0, 0] / Z  # Assuming fx is used for scaling (can adjust this formula based on actual focal length and depth behavior)
            scale_factor_h = self.K[1, 1] / Z
            scaled_width = int(landmark_widths[i] * scale_factor_w)
            scaled_height = int(landmark_heights[i] * scale_factor_h)

            # Optionally add class labels next to the bounding box
            if 0 <= x < self.img_width and 0 <= y < self.img_height:
                color = class_to_color[i]
                # Calculate top-left and bottom-right corners
                top_left = (x - scaled_width // 2, y - scaled_height // 2)
                bottom_right = (x + scaled_width // 2, y + scaled_height // 2)

                # Draw the bounding box
                pre_projected = projected_image.copy()
                cv2.rectangle(projected_image, (x-10,y-10), (x+10, y+10), (255,255,255), -1)
                cv2.rectangle(projected_image, top_left, bottom_right, color, -1)  # Green box

                tx = x if x > 10 else 10
                tx = x if x < self.img_width - 10 else self.img_width - 10
                ty = y if y > 20 else 20
                ty = y if y < self.img_height - 20 else self.img_height - 20

                cv2.putText(projected_image, str(i),(tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                             (0, 0, 0), 2)
                alpha =0.15
                projected_image = cv2.addWeighted(projected_image, alpha, pre_projected, 1 - alpha, 0, pre_projected)
                #cv2.circle(projected_image, (x, y), 4, color, -1)  # Green dot
                # cv2.putText(projected_image, landmark_classes[i], (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                #             (255, 255, 255), 1)

                obj_dic = {"label": landmark_classes[i],
                           "x": x,
                           "y": y,
                           "z": Z,
                           "i": i
                           }
                obj.append(obj_dic)


        self.vlm_cls_input = [d["label"] for d in obj]
        self.vlm_cls_input_num = [d["i"] for d in obj]
        #print(self.vlm_cls_input)

        if self.save_projections:
            self.frame_num += 1
            json_out["image_idx"] = "{:05d}_ori.png".format(self.frame_num)
            json_out["contents"] = obj

            self.save_img(img, projected_image)
            self.save_json(json_out)

        self.vlm_img_input = img
        return projected_image

    def save_img(self, img, projected_image):

        output_path = self.output_dir  # / time_string
        output_path.mkdir(parents=True, exist_ok=True)

        _output_path = output_path / "{:05d}.png".format(self.frame_num)
        cv2.imwrite(str(_output_path), img)

        _output_path = output_path / "{:05d}_proj.png".format(self.frame_num)
        cv2.imwrite(str(_output_path), projected_image)

    def save_json(self, json_out):
        import json
        name = json_out["image_idx"][:-4] + ".json"
        with open(self.output_dir / name, "w") as f:
            json.dump(json_out, f, indent=4)

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

    def get_class_index(self, class_name):
        # Returns the index of the specified class in the class list
        try:
            self.classlist = self.classstring.split(", ")
            out = []
            for cname in class_name:
                # print(f"Class '{cname}' found at index {self.classlist.index(cname)}")
                out.append(self.classlist.index(cname))
            return out  # self.classlist.index(class_name)
        except ValueError:
            print(f"Class '{class_name}' not found in class list.")
            return [-1]  # Returns -1 if the class is not found

    def get_model_output(self):

        if self.classstring == "":
            return [-1]

        if self.vlm_cls_input == []:
            print(self.vlm_cls_input)
            return [-1]

        print("frame : ", self.frame_num)
        #self.frame_num += 1  # increase frame number saving only
        print("tags :", self.vlm_cls_input, self.vlm_cls_input_num)

        txt_input = f"tags = {self.vlm_cls_input}\ntag_numbers = {self.vlm_cls_input_num}"

        self.vlm_filter.reset_memory()
        vlm_response = self.vlm_filter.call_vision_agent_with_image_input(self.vlm_img_input, txt_input, self.client)
        str_response = return_str(vlm_response)

        # Extract the part of the string that represents the list
        list_from_string = str_response.split('=')[-1].strip()
        try:
            list_from_string = literal_eval(list_from_string)
        except:
            # import re
            # list_from_string = re.sub(r'(\w+)', r'"\1"', list_from_string)
            # list_from_string = literal_eval(list_from_string)
            print(f"remove : {list_from_string}")
            list_from_string = [0]

        print("remove : ", list_from_string)
        self.vlm_cls_input = []

        return self.get_class_index(list_from_string)

    def call_remove_class_service(self, class_id):
        rospy.wait_for_service('remove_class')
        try:
            out = []
            for id in class_id:
                remove_class = rospy.ServiceProxy('remove_class', RemoveClass)
                req = RemoveClassRequest(class_id=id)
                res = remove_class(req)
                out.append(res.success)
            return -1 if False in out else 1  # res.success
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
            return False


if __name__ == "__main__":
    rospy.init_node("landmarks_comparison_and_removal")
    detector = Compare2DMapAndImage()

    while not rospy.is_shutdown():
        class_id = detector.get_model_output()
        rospy.loginfo(f"Calling service to remove class ID: {class_id}")
        success = detector.call_remove_class_service(class_id)
        rospy.loginfo("Service call success: %s" % success)
        ## change this time if you want to change the frequency of the service call
        rospy.sleep(7)  # Simulate processing time 10
