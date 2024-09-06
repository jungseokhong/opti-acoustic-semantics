#!/usr/bin/env python3
from dotenv import load_dotenv

from descriptive_tag_generator import tag_generator

load_dotenv()

import os
from pathlib import Path
import json

import message_filters
import numpy as np
import rospy
from cv_bridge import CvBridge
import cv2
from openai import OpenAI
from semanticslam_ros.msg import MapInfo, ObjectsVector, AllClassProbabilities, ClassProbabilities

from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as RosImage
from scipy.spatial.transform import Rotation as R
from semanticslam_ros.srv import RemoveClass, RemoveClassRequest
from semanticslam_ros.srv import RemoveLandmark, RemoveLandmarkRequest
from semanticslam_ros.srv import ModifyLandmark, ModifyLandmarkRequest
import asyncio

from ast import literal_eval

from vlm_filter_utils import vision_filter

from collections import defaultdict

import sys

sys.path.append(os.environ['LLM_MAPPING'])

import random
from beantown_agent.map_agent import vision_agent
from beantown_agent.agent_utils import return_str


# K: [527.150146484375, 0.0, 485.47442626953125, 0.0, 527.150146484375, 271.170166015625, 0.0, 0.0, 1.0]
# [TODO] should subscribe to the camera info topic to get the camera matrix K rather than hardcoding it


def generate_unique_colors(num_colors):
    hsv_colors = [(i * 180 // num_colors, 255, 255) for i in
                  range(num_colors)]  # Generate colors with maximum saturation and value
    bgr_colors = [cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0][0] for hsv in hsv_colors]
    colors = [tuple(color.tolist()) for color in bgr_colors]  # Convert each color from numpy array to tuple
    random.shuffle(colors)
    return colors


MODIFY_FUNCTION = False
REMOVE_DUPLICATES = True

class Compare2DMapAndImage:
    def __init__(self):

        self.save_projections = True
        self.output_dir = Path(os.environ['DATASETS']) / "llm_data/llm_filter_output"
        # self.delete_file(self.output_dir / "results.json")  # remove .json file if exist

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
        self.landmark_keys = []  # stores keys to remove landmarks
        self.landmark_keys_to_modify = []  # stores keys to modify landmarks
        self.newclasses_for_landmarks = []  # stores new classes for landmarks

        self.num_classes = 20  # Example: Specify the number of classes
        self.colors = generate_unique_colors(self.num_classes)  # Generate unique colors for these classes

        self.bridge = CvBridge()
        self.yoloimg_sub = message_filters.Subscriber('/camera/yolo_img', RosImage)
        self.rgb_sub = message_filters.Subscriber('/zed2i/zed_node/rgb/image_rect_color', RosImage)
        self.mapinfo_sub = message_filters.Subscriber('/mapinfo', MapInfo)
        self.classlist_sub = message_filters.Subscriber('/camera/objects', ObjectsVector)

        # Separate subscriber for CameraInfo
        self.cam_info_sub = rospy.Subscriber('/zed2i/zed_node/rgb/camera_info', CameraInfo, self.camera_info_callback)

        self.compare_pub = rospy.Publisher("/compareresults", RosImage, queue_size=10)
        self.allclsprobs_pub = rospy.Publisher('/allclass_probabilities', AllClassProbabilities, queue_size=10)

        # Use a cache for the mapinfo_sub to store messages
        self.mapinfo_cache = message_filters.Cache(self.mapinfo_sub, 500)
        self.rgb_cache = message_filters.Cache(self.rgb_sub, 500)
        self.yoloimg_cache = message_filters.Cache(self.yoloimg_sub, 500)
        self.classlist_cache = message_filters.Cache(self.classlist_sub, 500)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            (self.rgb_sub, self.yoloimg_sub, self.mapinfo_cache, self.classlist_cache), 500, 0.005
        )  # 0.025 need to reduce this time difference
        # # need to update so it can handle time offset/pass time offset

        self.sync.registerCallback(self.forward_pass)

        self.frame_num = 0
        self.client = OpenAI()
        self.tag_filter_api = vision_agent(vision_filter)
        self.tag_generator_api = vision_agent(tag_generator)

        self.descriptive_tag_json = Path.cwd() / "descriptive_tags"
        self.descriptive_tag_json.mkdir(exist_ok=True)
        self.descriptive_tag_json = self.descriptive_tag_json / "descriptive_tags.json"
        self.delete_file(self.descriptive_tag_json)

        # Initialize confusion matrix as a defaultdict of dicts
        self.confusion_matrix = defaultdict(lambda: defaultdict(int))
        self.confusion_matrix_for_duplicates = defaultdict(lambda: defaultdict(int))
        self.probabilities = defaultdict(dict)
        self.unique_precise_tags_list = []
        self.duplicate_tags_list = []
        self.landmark_keys_duplicated = []  # stores keys to remove landmarks (duplicating tags)


    def delete_file(self, path):
        if path.exists():
            path.unlink(missing_ok=True)

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

    def forward_pass(self, rgbimg: RosImage, yoloimg: RosImage, map_info: MapInfo, objects_info: ObjectsVector) -> None:

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
        rgbimg_cv = self.bridge.imgmsg_to_cv2(rgbimg, desired_encoding='bgr8')
        # print(map_info)

        # Extract data from map_info
        position, orientation, landmark_points, landmark_classes, landmark_widths, landmark_heights, landmark_keys = self.parse_data(
            map_info)
        # Project landmarks to the image
        projected_image = self.projectLandmarksToImage_removeoverlap(position, orientation, landmark_points,
                                                                     landmark_classes,
                                                                     landmark_widths, landmark_heights, landmark_keys,
                                                                     img=rgbimg_cv)

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

        ## [TODO]: this may not be the perfect place to publish all class probabilities

        allclass_probs = AllClassProbabilities()
        allclass_probs.header = rgbimg.header
        allclass_probs.classes = []

        for predicted_class, corrections in self.probabilities.items():
            classes_prob = ClassProbabilities()
            classes_prob.predicted_class = predicted_class
            classes_prob.corrected_classes = list(corrections.keys())
            classes_prob.probabilities = list(corrections.values())
            allclass_probs.classes.append(classes_prob)
            # print(f'length of allclass_probs.classes: {len(allclass_probs.classes)}')

        # Publish all class probabilities
        self.allclsprobs_pub.publish(allclass_probs)
        # rospy.loginfo(f"Publishing probabilities for all classes")

    def parse_data(self, map_info):

        landmark_points = []
        landmark_classes = []
        position_data = []
        orientation_data = []
        landmark_widths = []
        landmark_heights = []
        landmark_keys = []

        num_landmarks = len(map_info.landmark_points)
        for i in range(num_landmarks):
            landmark_points.append(map_info.landmark_points[i].x)
            landmark_points.append(map_info.landmark_points[i].y)
            landmark_points.append(map_info.landmark_points[i].z)
            landmark_classes.append(map_info.landmark_classes[i])
            landmark_widths.append(map_info.landmark_widths[i])
            landmark_heights.append(map_info.landmark_heights[i])
            landmark_keys.append(map_info.landmark_keys[i])

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
        landmark_keys_array = np.array(landmark_keys)
        # Reorder the orientation array
        # orientation_array = np.roll(orientation_array, -3) # for w,x,y,z convention

        return position_array, orientation_array, landmark_points_array, landmark_classes_array, landmark_widths_array, landmark_heights_array, landmark_keys_array

    def draw_dashed_rectangle(self, img, pt1, pt2, color=(0, 0, 0), thickness=1, dash_length=1):
        x1, y1 = pt1
        x2, y2 = pt2
        # hor
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
            cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)

        # ver
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
            cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)

    def tag_edge(self, edge, tlx, tly, brx, bry, box_size, add_w):

        if edge == "tl_i":
            tlp_y = tly  # inside
            brp_y = tly + box_size
            tlp_x = tlx
            brp_x = tlx + box_size + add_w
        elif edge == "tl_o":
            if tly - box_size < 0:
                tlp_y = tly  # inside
                brp_y = tly + box_size
            else:
                tlp_y = tly - box_size
                brp_y = tly
            tlp_x = tlx
            brp_x = tlx + box_size + add_w
        elif edge == "lt_o":  # left-top
            if tlx - box_size - add_w < 0:
                tlp_x = tlx  # inside
                brp_x = tlx + box_size + add_w
            else:
                tlp_x = tlx - box_size - add_w
                brp_x = tlx
            tlp_y = tly
            brp_y = tly + box_size
        elif edge == "rt_i":
            tlp_x = brx - box_size - add_w  # inside
            brp_x = brx
            tlp_y = tly
            brp_y = tly + box_size
        elif edge == "rt_o":
            if brx + box_size + add_w > self.img_width:
                tlp_x = brx - box_size - add_w  # inside
                brp_x = brx
            else:
                tlp_x = brx
                brp_x = brx + box_size + add_w
            tlp_y = tly
            brp_y = tly + box_size
        elif edge == "bl_i":
            tlp_y = bry - box_size  # inside
            brp_y = bry
            tlp_x = tlx
            brp_x = tlx + box_size + add_w
        elif edge == "bl_o":
            if bry + box_size > self.img_height:
                tlp_y = bry - box_size  # inside
                brp_y = bry
            else:
                tlp_y = bry
                brp_y = bry + box_size
            tlp_x = tlx
            brp_x = tlx + box_size + add_w

        elif edge == "br_i":
            tlp_y = bry - box_size  # inside
            brp_y = bry
            tlp_x = brx - box_size - add_w
            brp_x = brx
        else:  # edge == "br_o":
            if bry + box_size > self.img_height:
                tlp_y = bry - box_size  # inside
                brp_y = bry
            else:
                tlp_y = bry
                brp_y = bry + box_size
            tlp_x = brx - box_size - add_w
            brp_x = brx

        return ((tlp_x, tlp_y), (brp_x, brp_y))

    def tag_box(self, tlx, tly, brx, bry, box_size, tag_area):
        add_w = int(box_size // 1.3)
        edge = ["tl_i", "tl_o", "lt_o", "rt_i", "rt_o", "bl_i", "bl_o", "br_i", "br_o", "tl_i"]

        for edge_mode in edge:
            (tlp_x, tlp_y), (brp_x, brp_y) = self.tag_edge(edge_mode, tlx, tly, brx, bry, box_size, add_w)
            if tag_area == []:
                tag_area.append((tlp_x, tlp_y, brp_x, brp_y))
                return ((tlp_x, tlp_y), (brp_x, brp_y))
            overlapping = False
            for area in tag_area:
                if (area[0] <= brp_x and area[2] >= tlp_x) and (area[1] <= brp_y and area[3] >= tlp_y):
                    overlapping = True
                    break
            if not overlapping:
                tag_area.append((tlp_x, tlp_y, brp_x, brp_y))
                return ((tlp_x, tlp_y), (brp_x, brp_y))

        tag_area.append((tlp_x, tlp_y, brp_x, brp_y))
        return ((tlp_x, tlp_y), (brp_x, brp_y))

    def projectLandmarksToImage(self, position, orientation, landmark_points, landmark_classes, landmark_widths,
                                landmark_heights, landmark_keys, img=None):

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

        # project points in world frame to camera frame
        RT_2 = np.eye(4)
        RT_2[:3, :3] = R_C_W
        RT_2[:3, 3] = -R_C_W @ t_vec
        points_3d_homo_camera_frame = RT_2 @ homogeneous_points
        # print(f'points_3d_homo_camera_frame shape {points_3d_homo_camera_frame.shape}, and real data:  {points_3d_homo_camera_frame}')

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
        tag_area = []

        self.vlm_img_input = []
        self.vlm_img_input.append(img.copy())
        for i, (point_3d, landmark_key, point_2d, point_3d_camera_frame) in enumerate(
                zip(landmark_points, landmark_keys, points_2d_homo.T, points_3d_homo_camera_frame.T)):

            x, y = int(point_2d[0] / point_2d[2]), int(point_2d[1] / point_2d[2])
            # Compute the Euclidean distance between the point and the camera position
            Z = np.linalg.norm(point_3d - position)

            # Pass if the landmark z location is behind the camera's location
            # Project landmarks only less than 5m away from current position. need better algorithm to filter this.
            if point_3d_camera_frame[2] < 0 or Z >= 5:
                continue

            # Scale widths and heights based on the depth
            scale_factor_w = self.K[0, 0] / Z
            scale_factor_h = self.K[1, 1] / Z
            scaled_width = int(landmark_widths[i] * scale_factor_w)
            scaled_height = int(landmark_heights[i] * scale_factor_h)

            # Optionally add class labels next to the bounding box
            if 0 <= x < self.img_width and 0 <= y < self.img_height:
                color = class_to_color[i]

                # increse bbox size + 5%
                add_w = int((scaled_width * 1.05) // 2)
                add_h = int((scaled_height * 1.05) // 2)

                # Calculate top-left and bottom-right corners
                tlx = (x - scaled_width // 2) - add_w if (x - scaled_width // 2) - add_w >= 0 else 1
                tly = (y - scaled_height // 2) - add_h if (y - scaled_height // 2) - add_h >= 0 else 1
                brx = (x + scaled_width // 2) + add_w if (
                                                                 x + scaled_width // 2) + add_w < self.img_width else self.img_width - 2
                bry = (y + scaled_height // 2) + add_h if (
                                                                  y + scaled_height // 2) + add_h < self.img_height else self.img_height - 2

                ##addcrop an img
                self.vlm_img_input.append(img[tly:bry, tlx:brx])
                # frame_num_txt = "{:05d}".format(self.frame_num + 1)
                # cv2.imwrite(f"{self.output_dir}/{frame_num_txt}_{i}.png", img[tly:bry, tlx:brx])

                ##drawing
                # tag_box
                b_size = 25
                tag_tl, tag_br = self.tag_box(tlx, tly, brx, bry, b_size, tag_area)

                # Draw the bounding box
                pre_projected = projected_image.copy()
                cv2.rectangle(projected_image, tag_tl, tag_br, color, -1)  # tag
                cv2.rectangle(projected_image, (tlx, tly), (brx, bry), color, 3)  # boudingbox 1 - 0.45

                alpha = 0.4
                projected_image = cv2.addWeighted(projected_image, alpha, pre_projected, 1 - alpha, 0, pre_projected)

                cv2.rectangle(projected_image, tag_tl, tag_br, color, 1)  # tag
                # cv2.rectangle(projected_image, (tlx, tly), (brx, bry), color, 1)
                self.draw_dashed_rectangle(projected_image, (tlx, tly), (brx, bry), color, dash_length=5)

                # tag_name = f"[{i}]"
                # cv2.putText(projected_image, tag_name, (tag_tl[0], tag_br[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                #            (0, 0, 0), 1)
                # cv2.rectangle(projected_image, (tlx, tly), (brx, bry), color, 1)

                obj_dic = {"label": landmark_classes[i],
                           "x": x,
                           "y": y,
                           "z": Z,
                           "i": i,
                           "landmark_key": str(landmark_key)
                           }
                obj.append(obj_dic)

        # print text on the image
        for single_obj, tag in zip(obj, tag_area):
            tag_name = f"[{single_obj['i']}]"
            cv2.putText(projected_image, tag_name, (tag[0], tag[3] - 6), cv2.FONT_HERSHEY_TRIPLEX, 0.65,
                        (0, 0, 0), 1, cv2.LINE_AA)

        self.vlm_cls_key = [np.int64(d["landmark_key"]) for d in obj]  # key
        self.vlm_cls_input = [d["label"] for d in obj]  # class name
        self.vlm_cls_input_idx = [d["i"] for d in obj]  # index

        self.frame_num += 1
        if self.save_projections:
            json_out["image_idx"] = "{:05d}_ori.png".format(self.frame_num)
            # json_out["{:05d}.png".format(self.frame_num)] = {"contents": obj}
            json_out["contents"] = obj

            self.save_img(img, projected_image)
            self.save_json(json_out, self.output_dir)

        # self.vlm_img_input = projected_image
        self.vlm_img_input[0] = projected_image
        return projected_image

    def removeoverlap(self, bounding_boxes, depths, obj):
        overlapping_indices = []
        for i, (bbox1, depth1) in enumerate(zip(bounding_boxes, depths)):
            for j, (bbox2, depth2) in enumerate(zip(bounding_boxes[i + 1:], depths[i + 1:])):
                tl1, br1 = bbox1
                tl2, br2 = bbox2
                # opencv image coordinate system is top-left origin
                area1 = (br1[0] - tl1[0]) * (-tl1[1] + br1[1])
                area2 = (br2[0] - tl2[0]) * (-tl2[1] + br2[1])

                intersect_tl = (max(tl1[0], tl2[0]), max(tl1[1], tl2[1]))
                intersect_br = (min(br1[0], br2[0]), min(br1[1], br2[1]))

                intersect_area = max(0, intersect_br[0] - intersect_tl[0]) * max(0, -intersect_tl[1] + intersect_br[1])

                overlap = intersect_area / min(area1, area2)

                if overlap > 0.7 and abs(depth1 - depth2) > 5:
                    if depth1 > depth2:
                        overlapping_indices.append(i)
                    else:
                        overlapping_indices.append(i + j + 1)

        overlapping_indices = sorted(set(overlapping_indices), reverse=True)
        for idx in overlapping_indices:
            del bounding_boxes[idx]
            del depths[idx]
            del obj[idx]

    def removeoverlap_with_semantics(self, bounding_boxes, depths, obj, precise_tag_list):
        overlapping_indices = []
        
        for i, (bbox1, depth1, class1) in enumerate(zip(bounding_boxes, depths, obj)):
            if class1["label"] not in precise_tag_list:
                continue
            
            for j, (bbox2, depth2, class2) in enumerate(zip(bounding_boxes[i + 1:], depths[i + 1:], obj[i + 1:])):
                tl1, br1 = bbox1
                tl2, br2 = bbox2
                
                # print(f'tl1: {tl1}, br1: {br1}, tl2: {tl2}, br2: {br2} bbox1: {bbox1}, bbox2: {bbox2}')
                area1 = (br1[0] - tl1[0]) * (-(tl1[1] - br1[1]))
                area2 = (br2[0] - tl2[0]) * (-(tl2[1] - br2[1]))

                intersect_tl = (max(tl1[0], tl2[0]), max(tl1[1], tl2[1]))
                intersect_br = (min(br1[0], br2[0]), min(br1[1], br2[1]))

                intersect_area = max(0, intersect_br[0] - intersect_tl[0]) * max(0, -intersect_tl[1] + intersect_br[1])
                overlap = intersect_area / min(area1, area2)

                if overlap > 0.7 and abs(depth1 - depth2) < 0.5:
                # if True:
                    print(f"======overlap: {overlap} depth: {abs(depth1 - depth2)} Overlap detected between {class1['label']} and {class2['label']}")
                    class1_label = class1["label"]
                    class2_label = class2["label"]


                    # Check if class2 (less precise) is semantically connected to class1 (precise)
                    # print(f'class1_label: {class1_label}, in this : {list(self.confusion_matrix_for_duplicates[class2_label].keys())}')

                    # for duplicated_class, precise_corrections in self.confusion_matrix_for_duplicates.items():
                    #     print(f'precise_corrections keys: {precise_corrections.keys()}')
                    #     if class1_label in precise_corrections.keys():
                    #         print(f"Duplicate tag found: {duplicated_class} will be removed due to {class1_label}")
                    #         overlapping_indices.append(i + j + 1)
                    #         self.landmark_keys_duplicated.append(np.int64(class2["landmark_key"]))
                            
                    
                    
                    if class1_label in list(self.confusion_matrix_for_duplicates[class2_label].keys()):
                        print(f'====class1_label: {class1_label}, in this : {list(self.confusion_matrix_for_duplicates[class2_label].keys())}')
                        # If connected by semantics, mark the less precise bounding box for removal
                        overlapping_indices.append(i + j + 1)
                        self.landmark_keys_duplicated.append(np.int64(class2["landmark_key"]))
                        print(f"Duplicate tag found: {class2_label} will be removed due to {class1_label}")

        # Remove duplicates and reverse the list to safely delete without affecting indices
        overlapping_indices = sorted(set(overlapping_indices), reverse=True)
        for idx in overlapping_indices:
            del bounding_boxes[idx]
            del depths[idx]
            del obj[idx]

    def draw_tags_boxes(self, projected_image, bounding_boxes, obj,
                        class_to_color, alpha=0.4, tag_box_size=25, cropped_imgs=False):
        tag_area = []  # To determine if tags  are overlapping
        for ((tlx, tly), (brx, bry)), color in zip(bounding_boxes, class_to_color.values()):
            # get tag_box positions
            (tlp_x, tlp_y), (brp_x, brp_y) = self.tag_box(tlx, tly, brx, bry, tag_box_size, tag_area)

            pre_projected = projected_image.copy()
            cv2.rectangle(projected_image, (tlp_x, tlp_y), (brp_x, brp_y), color, -1)  # tag
            cv2.rectangle(projected_image, (tlx, tly), (brx, bry), color, 3)  # bounding box

            projected_image = cv2.addWeighted(projected_image, alpha, pre_projected, 1 - alpha, 0, pre_projected)

            cv2.rectangle(projected_image, (tlp_x, tlp_y), (brp_x, brp_y), color, 1)  # tags
            self.draw_dashed_rectangle(projected_image, (tlx, tly), (brx, bry), color, dash_length=5)

        # print text on the image
        for idx, (single_obj, tag) in enumerate(zip(obj, tag_area)):
            tag_name = f"[{single_obj['i']}]"
            cv2.putText(projected_image, tag_name, (tag[0], tag[3] - 6), cv2.FONT_HERSHEY_TRIPLEX, 0.65,
                        (0, 0, 0), 1, cv2.LINE_AA)

            if cropped_imgs:
                pre_projected = self.vlm_img_input[idx + 1].copy()
                add_w = int(tag_box_size // 1.3)
                cv2.rectangle(pre_projected, (0, 0), (tag_box_size + add_w, tag_box_size), class_to_color[idx], 1)
                cv2.rectangle(pre_projected, (0, 0), (tag_box_size + add_w, tag_box_size), class_to_color[idx], -1)
                self.vlm_img_input[idx + 1] = cv2.addWeighted(self.vlm_img_input[idx + 1], alpha, pre_projected,
                                                              1 - alpha, 0, pre_projected)
                cv2.putText(self.vlm_img_input[idx + 1], tag_name, (0, tag_box_size - 6), cv2.FONT_HERSHEY_TRIPLEX,
                            0.65,
                            (0, 0, 0), 1, cv2.LINE_AA)

        return projected_image

    def projectLandmarksToImage_removeoverlap(self, position, orientation,
                                              landmark_points, landmark_classes,
                                              landmark_widths, landmark_heights, landmark_keys,
                                              img=None):
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

        # project points in world frame to camera frame
        RT_2 = np.eye(4)
        RT_2[:3, :3] = R_C_W
        RT_2[:3, 3] = -R_C_W @ t_vec
        points_3d_homo_camera_frame = RT_2 @ homogeneous_points
        # print(f'points_3d_homo_camera_frame shape {points_3d_homo_camera_frame.shape}, and real data:  {points_3d_homo_camera_frame}')

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
        tag_area = []
        bounding_boxes = []
        depths = []

        self.vlm_img_input = []
        self.vlm_img_input.append(img.copy())

        for i, (point_3d, landmark_key, point_2d, point_3d_camera_frame) in enumerate(
                zip(landmark_points, landmark_keys, points_2d_homo.T, points_3d_homo_camera_frame.T)):
            x, y = int(point_2d[0] / point_2d[2]), int(point_2d[1] / point_2d[2])
            # Compute the Euclidean distance between the point and the camera position
            Z = np.linalg.norm(point_3d - position)

            # Pass if the landmark z location is behind the camera's location
            # Project landmarks only less than 5m away from current position. need better algorithm to filter this.
            if point_3d_camera_frame[2] < 0 or Z >= 5:
                continue

            # Scale widths and heights based on the depth
            scale_factor_w = self.K[0, 0] / Z
            scale_factor_h = self.K[1, 1] / Z
            scaled_width = int(landmark_widths[i] * scale_factor_w)
            scaled_height = int(landmark_heights[i] * scale_factor_h)

            if 0 <= x < self.img_width and 0 <= y < self.img_height:
                # padding  bbox size + 5%
                add_w = int((scaled_width * 1.05) // 2)
                add_h = int((scaled_height * 1.05) // 2)

                # get valid bounding box positions
                tlx = (x - scaled_width // 2) - add_w if (x - scaled_width // 2) - add_w >= 0 else 1
                tly = (y - scaled_height // 2) - add_h if (y - scaled_height // 2) - add_h >= 0 else 1
                brx = (x + scaled_width // 2) + add_w if (x + scaled_width // 2) + add_w < self.img_width \
                    else self.img_width - 2
                bry = (y + scaled_height // 2) + add_h if (y + scaled_height // 2) + add_h < self.img_height \
                    else self.img_height - 2

                ##add cropped images
                self.vlm_img_input.append(img[tly:bry, tlx:brx].copy())

                bounding_boxes.append(((tlx, tly), (brx, bry)))
                depths.append(Z)

                # save inf
                obj_dic = {"label": landmark_classes[i],
                           "x": x, "y": y, "z": Z, "i": i,
                           "landmark_key": str(landmark_key)}
                obj.append(obj_dic)

        self.removeoverlap(bounding_boxes, depths, obj)
        self.removeoverlap_with_semantics(bounding_boxes, depths, obj, self.unique_precise_tags_list)
        projected_image = self.draw_tags_boxes(projected_image, bounding_boxes,
                                               obj, class_to_color, alpha=0.4, tag_box_size=25, cropped_imgs=True)

        self.vlm_cls_key = [np.int64(d["landmark_key"]) for d in obj]  # key
        self.vlm_cls_input = [d["label"] for d in obj]  # class name
        self.vlm_cls_input_idx = [d["i"] for d in obj]  # index
        self.vlm_cls_location = [[d["x"], d["y"], d["z"]] for d in obj]

        self.frame_num += 1
        if self.save_projections:
            json_out["image_idx"] = "{:05d}_ori.png".format(self.frame_num)
            # json_out["{:05d}.png".format(self.frame_num)] = {"contents": obj}
            json_out["contents"] = obj

            self.save_img(img, projected_image)
            self.save_json(json_out, self.output_dir)

        # self.vlm_img_input = projected_image
        self.vlm_img_input[0] = projected_image
        return projected_image

    def save_img(self, img, projected_image):
        output_path = self.output_dir  # / time_string
        output_path.mkdir(parents=True, exist_ok=True)

        _output_path = output_path / "{:05d}.png".format(self.frame_num)
        cv2.imwrite(str(_output_path), img)

        _output_path = output_path / "{:05d}_proj.png".format(self.frame_num)
        cv2.imwrite(str(_output_path), projected_image)

    def save_json(self, json_out, output_dir):
        json_path = json_out["image_idx"][:-4] + ".json"
        json_path = output_dir / json_path

        with open(json_path, "w") as f:
            json.dump(json_out, f, indent=4)

    def save_json_from_path(self, json_path, conts) -> None:
        with open(json_path, "w") as f:
            json.dump(conts, f, indent=4)

    def open_json(self, json_path) -> dict:
        if Path(json_path).exists():
            with open(json_path, 'r') as file:
                data = json.load(file)
        else:
            data = {}
        return data

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

    def call_api_with_img(self, vlm_filter, vlm_img_input, txt_input):
        vlm_response = vlm_filter.call_vision_agent_with_image_input(imgs=vlm_img_input, command=txt_input,
                                                                     client=self.client)
        str_response = return_str(vlm_response)
        return str_response

    def call_api(self, vlm_filter, txt_input):
        vlm_response = vlm_filter.call_vision_agent(txt_input, self.client)
        str_response = return_str(vlm_response)
        return str_response

    def memory_injection(self, vlm_filter):
        user_commant = """cup: 0, book: 1, baseball hat: 3, baseball hat: 4, hat: 7"""
        commant = """\nExamples of each step's output for the given image and its tags:\n"""
        assistant_commant = """Step 1. 
        Tag 0 (cup): Incorrect.The bounding box contains a ball.
        Tag 1 (book): Correct. 
        Tag 3 (baseball hat): Correct. 
        Tag 4 (baseball hat): Correct.   
        Tag 7 (hat): Correct. 
    Step 2. 
        Tags [3, 4, 7] are pointing to the same object. 
    Step 3. 
        Tags [3, 4, 7] : "Baseball hat" is a more precise tag than "hat" since there is an LA mark on it. Tag 3 focuses on only a smaller part, but Tag 4 covers the entire object. Therefore, precise_tag = [4]
    Step 4. 
        unmatched_tags = [0]
        unmatched_tags = [3, 7]
    Step 5.
        unmatched_tags = [0, 3, 7] """

        vlm_filter.reset_with_img(role="user", prompt=user_commant,
                                  img="/home/beantown/ran/llm_ws/src/maxmixtures/opti-acoustic-semantics/example_image.png")
        vlm_filter.add_memory_with_prompts(role="assistant", prompt=assistant_commant)

        ####
        # vlm_filter.reset_with_img(role="system", prompt=user_commant+commant+assistant_commant,
        #                                img="/home/beantown/ran/llm_ws/src/maxmixtures/opti-acoustic-semantics/example_image.png")
        #

    def extract_list(self, response, keyword):
        rest_response = response.split(keyword)[-1].split('[', 1)[-1].strip()
        part = rest_response.split(']')[0]
        extracted_list = '[' + part + ']'
        extracted_list = extracted_list.replace("<", "").replace(">", "")
        extracted_list = literal_eval(extracted_list)
        return extracted_list, rest_response

    def return_landmarks_to_remove(self, str_response, vlm_cls_input, vlm_cls_input_idx):
        ## Extract the part of the string that represents the list

        try:
            empty_tags, str_response = self.extract_list(str_response, 'empty_tags')
        except:
            empty_tags = []
        try:
            incorrect_tags, str_response = self.extract_list(str_response, 'incorrect_tags')
        except:
            incorrect_tags = []
        try:
            corrected_tags, str_response = self.extract_list(str_response, 'corrected_tags')
        except:
            corrected_tags = []
        try:
            duplicated_tags, str_response = self.extract_list(str_response, 'duplicated_tags')
        except:
            duplicated_tags = []
        try:
            precise_tags, str_response = self.extract_list(str_response, 'precise_tags_in_duplicated_tags')
        except:
            precise_tags = []

        # less_precise = duplicated_tags - precise_tags
        less_precise = [item for sublist in duplicated_tags for item in sublist if item not in precise_tags]
        tag_idx_to_remove = empty_tags + incorrect_tags + less_precise

        # replace a cls id to a cls name
        tags_to_remove = [vlm_cls_input[vlm_cls_input_idx.index(int(i))] for i in
                          tag_idx_to_remove]

        return tags_to_remove, tag_idx_to_remove, (incorrect_tags, corrected_tags), (duplicated_tags, precise_tags)

    def return_descriptive_tags(self, str_response, cls_names, cls_idc, cls_keys, cls_locations) -> None:
        cls_matched_descriptive_tags = []
        cls_idx_matched_descriptive_tags = []
        cls_key_matched_descriptive_tags = []
        cls_key_matched_locations = []
        parts = str_response.split('tag_')

        for part in parts[1:]:
            part = part.replace("*", " ")
            tag_list = part.split('=')
            cls_idx = int(tag_list[0].strip())
            tag_list = tag_list[1].split('\']')[0] + '\']'
            try:
                base = np.where(np.isin(cls_idc, cls_idx))[0][0]
                tag_list = literal_eval(tag_list.strip())

                cls_matched_descriptive_tags.append(cls_names[base])
                cls_idx_matched_descriptive_tags.append(tag_list)
                cls_key_matched_descriptive_tags.append(cls_keys[base])
                cls_key_matched_locations.append(cls_locations[base])
                print("generated tags:", tag_list)

            except:
                print(f"\nerror during extract the result list..{part}\n")

        if len(cls_matched_descriptive_tags) < 1:
            return []

        # save json file
        data = self.open_json(self.descriptive_tag_json)

        for general_tag, descriptive_tags, general_tag_key, location in zip(cls_matched_descriptive_tags,
                                                                            cls_idx_matched_descriptive_tags,
                                                                            cls_key_matched_descriptive_tags,
                                                                            cls_key_matched_locations):

            info = {int(general_tag_key): {"features": descriptive_tags, "location": location}}
            if general_tag not in data:
                data[general_tag] = {}
            if f'{descriptive_tags[0]}' not in data[general_tag]:
                data[general_tag][f'{descriptive_tags[0]}'] = info

        self.save_json_from_path(self.descriptive_tag_json, data)
        return tag_list

    ## TODO: finalize this function
    def return_landmarks_to_modify(self, str_response, vlm_cls_input, vlm_cls_input_idx):
        ## Extract the part of the string that represents the list
        # list_from_idx = str_response.split('=')[-1].strip()
        ## hopefully, str_response has a idx for each landmark and what to change it do.
        ## e.g. [1] -> "red book"
        # return 1: "red book", 3 : "blue cup"

        # try:
        #     list_from_idx = literal_eval(list_from_idx)
        # except:
        #     print(f"\nerror during extract the result list..{list_from_idx}\n")
        #     list_from_idx = []  # prevent the program from stopping

        # # replace a cls id to a cls name
        # list_from_string = [vlm_cls_input[vlm_cls_input_idx.index(int(i))] for i in
        #                     list_from_idx]
        # modify_dic = {1: "red book", 0: "blue cup"}
        modify_dic = {0: "blue cup"}

        return modify_dic

    def update_confusion_matrix(self, predicted_class, corrected_class):
        """
        Update the confusion matrix with the predicted and corrected class.

        Args:
            predicted_class (str): The class predicted by the object detector.
            corrected_class (str): The class corrected by the VLM.
        """
        # Increment the count for the corrected class under the predicted class
        self.confusion_matrix[predicted_class][corrected_class] += 1

    def update_confusion_matrix_for_duplicates(self, predicted_class, corrected_class):
        """
        Update the confusion matrix with the predicted and corrected class.

        Args:
            predicted_class (str): The class predicted by the object detector.
            corrected_class (str): The class corrected by the VLM.
        """
        # Increment the count for the corrected class under the predicted class
        self.confusion_matrix_for_duplicates[predicted_class][corrected_class] += 1

    def calculate_probabilities(self):
        """
        Calculate and print the probability of each predicted class being corrected to each other class.
        """
        # probabilities = defaultdict(dict)
        for predicted_class, corrections in self.confusion_matrix.items():
            total_predictions = sum(corrections.values())
            for corrected_class, count in corrections.items():
                self.probabilities[predicted_class][corrected_class] = count / total_predictions

        return True  # self.probabilities

    def save_filter_response_json(self, frame_num, output_dir, json_name,
                                  filter_prompts, filter_txt_input, filter_str_response, filtered_tags):

        ##save it to json file
        # json_path = output_dir / "results.json"
        json_path = output_dir / json_name
        if json_path.exists():
            f = open(json_path)
            data = json.load(f)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            data = {"filter_prompts": filter_prompts.split('\n'),
                    "{:05d}.png".format(frame_num): {}
                    }

        # todo : remove this part? no need to write again
        if "filter_prompts" not in data:
            new_data = {}
            new_data["filter_prompts"] = filter_prompts.split('\n')
            for key in data:
                new_data[key] = data[key]
            data = new_data

        if "{:05d}.png".format(frame_num) not in data:
            data["{:05d}.png".format(frame_num)] = {}

        if type(filtered_tags).__name__ == "list":
            json_out = {"filter_text_input": filter_txt_input, "filter_response": filter_str_response.split('\n'),
                        "filtered_out": filtered_tags}
        else:
            items_to_remove1, incorrect_tags, corrected_tags = filtered_tags
            json_out = {"filter_text_input": filter_txt_input, "filter_response": filter_str_response.split('\n'),
                        "filtered_out": items_to_remove1,
                        "incorrect_tag_idx": incorrect_tags, "corrected_tags": corrected_tags}

        data["{:05d}.png".format(frame_num)]["llm_filter"] = json_out

        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

    def save_descriptor_response_json(self, frame_num, output_dir, json_name,
                                      tg_prompts, tg_txt_input, tg_str_response, descriptive_tags):

        ##save it to json file
        # json_path = output_dir / "results.json"
        json_path = output_dir / json_name
        if json_path.exists():
            f = open(json_path)
            data = json.load(f)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            data = {"tag_generator_prompts": tg_prompts.split('\n'),
                    "{:05d}.png".format(frame_num): {}
                    }

        # todo : remove this part? no need to write again
        if "tg_prompts" not in data:
            new_data = {}
            new_data["tag_generator_prompts"] = tg_prompts.split('\n')
            for key in data:
                new_data[key] = data[key]
            data = new_data

        if "{:05d}.png".format(frame_num) not in data:
            data["{:05d}.png".format(frame_num)] = {}

        json_out = {"tg_text_input": tg_txt_input, "tg_response": tg_str_response.split('\n'),
                    "generated_tag": descriptive_tags}
        data["{:05d}.png".format(frame_num)]["llm_tagger"] = json_out

        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

    async def tag_generator(self, frame_num, vlm_img_input,
                            vlm_cls_input, vlm_cls_input_idx, vlm_cls_key, vlm_cls_location,
                            idx_to_remove1):

        ## generate descriptive tags
        exist_descriptive_tags = self.open_json(self.descriptive_tag_json)
        no_need_description = []

        # skip it if the key is existed
        all_keys = [int(ssub_key) for par_key in exist_descriptive_tags.keys()
                    for sub_key in exist_descriptive_tags[par_key].keys() for ssub_key in
                    exist_descriptive_tags[par_key][sub_key].keys()]

        if len(all_keys) < 1:
            no_need_description = []
        else:
            no_need_description = np.where(np.isin(vlm_cls_key, all_keys))[0]
            # if the key is already existed in the descriptive tag list # might not need this part
            # no_need_description = np.array([])

            # if the tag is already descriptive
            all_tags = [[sub_key, par_key] for par_key in exist_descriptive_tags.keys() for sub_key in
                        exist_descriptive_tags[par_key].keys()]
            sub_tags = np.array(all_tags)[:, 0]

            for idx, (cls_name, cls_key) in enumerate(zip(vlm_cls_input, vlm_cls_key)):
                if idx not in no_need_description and cls_name in sub_tags:
                    no_need_description = np.append(no_need_description, idx)
                    tag_idx = np.where(sub_tags == cls_name)[0]
                    exist_descriptive_tags[all_tags[tag_idx[0]][1]][sub_tags[tag_idx[0]]][int(vlm_cls_key[idx])] = {}  #
            self.save_json_from_path(self.descriptive_tag_json, exist_descriptive_tags)  # add new tags

        tg_txt_input = []
        tag_num = []
        for idx, (cls_name, cls_num) in enumerate(zip(vlm_cls_input, vlm_cls_input_idx)):
            if cls_num in idx_to_remove1 or idx in no_need_description:
                continue
            tg_txt_input.append(f"tag {cls_num}-{cls_name}")
            tag_num.append(cls_num)
        tg_txt_input = ", ".join(tg_txt_input)
        tg_txt_input = f"Identify the features of only tags : {tg_txt_input}"

        if len(tag_num) < 1:
            print("No landmarks to create a descriptive tag")
            tag_api_response = "No new landmark to create a descriptive tag"
            generated_tags = []
        else:
            print(f"frame_num :{frame_num}, Descriptor : {tg_txt_input}")
            self.tag_generator_api.reset_memory()
            tag_api_response = self.call_api_with_img(self.tag_generator_api, vlm_img_input, tg_txt_input)
            generated_tags = self.return_descriptive_tags(tag_api_response, vlm_cls_input, vlm_cls_input_idx,
                                                          vlm_cls_key, vlm_cls_location)

        self.save_descriptor_response_json(frame_num, self.output_dir, "results1.json",
                                           self.tag_generator_api.args.system_prompt, tg_txt_input, tag_api_response,
                                           generated_tags)

    def get_model_output(self):
        if self.frame_num < 1:
            return True

        if len(self.vlm_cls_input) < 1:
            print("no landmark")
            return True

        # copy necessary vars
        frame_num = self.frame_num
        vlm_cls_input_idx = self.vlm_cls_input_idx
        vlm_cls_input = self.vlm_cls_input
        vlm_cls_key = self.vlm_cls_key
        vlm_img_input = self.vlm_img_input.copy()
        vlm_cls_location = self.vlm_cls_location

        print("frame : ", frame_num)
        print("tags :", vlm_cls_input, vlm_cls_input_idx)

        # for idx, img in enumerate(vlm_img_input[1:]):
        #     cv2.imwrite(str(self.output_dir / "{:05d}_input_{:02d}.png".format(frame_num, idx)), img)

        ### without cropped imgs
        # vlm_img_input = vlm_img_input[0] # without cropped imgs
        # cv2.imwrite(str(self.output_dir / "{:05d}_input.png".format(frame_num)), vlm_img_input)
        ### with cropped imgs
        cv2.imwrite(str(self.output_dir / "{:05d}_input.png".format(frame_num)), vlm_img_input[0])  # save an input img
        ###

        self.vlm_cls_input = []  # reset it, in order to prevent calling vlm repeatedly with the same input

        #### filtering api
        # edit a text input
        filter_txt_input = []
        tag_num = []
        for cls_name, cls_num in zip(vlm_cls_input, vlm_cls_input_idx):
            filter_txt_input.append(f"{cls_num}: {cls_name}")
            tag_num.append(cls_num)
        filter_txt_input = ", ".join(filter_txt_input)

        print(f"{filter_txt_input}")

        ## call api for filtering
        self.tag_filter_api.reset_memory()  # remove memorise
        str_response1 = self.call_api_with_img(self.tag_filter_api, vlm_img_input, filter_txt_input)

        items_to_remove1, idx_to_remove1, (incorrect_tags, corrected_tags), (
        duplicated_tags, precise_tags) = self.return_landmarks_to_remove(str_response1, vlm_cls_input,
                                                                         vlm_cls_input_idx)

        print(duplicated_tags, precise_tags)
        # call api to generating descriptive tags
        asyncio.run(self.tag_generator(frame_num, vlm_img_input,
                                       vlm_cls_input, vlm_cls_input_idx, vlm_cls_key, vlm_cls_location,
                                       idx_to_remove1))

        # print(f'vlm_cls_input_idx: {vlm_cls_input_idx} vlm_cls_input: {vlm_cls_input} vlm_cls_key: {vlm_cls_key}')
        # Creating the dictionary for vlm_cls_input
        vlm_cls_input_dict = dict(zip(vlm_cls_input_idx, vlm_cls_input))


        # Update the confusion matrix dynamically
        for k, (duplicated_tag_group, precise_tag) in enumerate(zip(duplicated_tags, precise_tags)):
            
            # Get the precise tag value from the dictionary
            precise_tag_value = vlm_cls_input_dict.get(precise_tag)

            # Ensure the precise tag value is unique
            if precise_tag_value not in self.unique_precise_tags_list:
                self.unique_precise_tags_list.append(precise_tag_value)

            for duplicated_tag in duplicated_tag_group:
                # Get the duplicate tag value from the dictionary
                duplicate_tag_value = vlm_cls_input_dict.get(duplicated_tag)
                if duplicate_tag_value != precise_tag_value:
                    self.update_confusion_matrix_for_duplicates(duplicate_tag_value, precise_tag_value)
                    # Add the duplicate tag to the list
                    if duplicate_tag_value not in self.duplicate_tags_list:
                        self.duplicate_tags_list.append(duplicate_tag_value)
        print("Unique Precise Tags:", self.unique_precise_tags_list)
        print("Duplicate Tags:", self.duplicate_tags_list)

        print("Confusion Matrix duplicate:")
        for predicted_class, corrections in self.confusion_matrix_for_duplicates.items():
            print(f"{predicted_class}: {dict(corrections)}")
        
        # Confusion Matrix duplicate:
        # fan: {'mechanical fan': 3}
        # mechanical fan: {'mechanical fan': 3}
        # toy: {'snowman': 4}
        # snowman: {'snowman': 4}
        # racket: {'tennis racket': 1}
        # tennis racket: {'tennis racket': 1}

        # confusion matrix for correct ones
        # Filtering correct tags by excluding incorrect_tags
        correct_tags = [idx for idx in vlm_cls_input_idx if idx not in incorrect_tags]
        # Filtering correct class names based on correct tags
        correct_cls_input = [cls_name for cls_name, idx in zip(vlm_cls_input, vlm_cls_input_idx) if idx not in incorrect_tags]
        print(f"correct_tags: {correct_tags} correct_cls_input: {correct_cls_input}")

        for correct_tag, original_correct_tag in zip(correct_tags, correct_cls_input):
            # if the tag has been removed then we don't need to update the confusion matrix
            if original_correct_tag == 'empty':
                continue
            self.update_confusion_matrix(vlm_cls_input_dict.get(correct_tag), original_correct_tag)
        # print("Confusion Matrix correct ones:")
        # for predicted_class, corrections in self.confusion_matrix.items():
        #     print(f"{predicted_class}: {dict(corrections)}")

        for incorrect_tag, corrected_tag in zip(incorrect_tags, corrected_tags):
            if corrected_tag == 'empty':
                continue
            self.update_confusion_matrix(vlm_cls_input_dict.get(incorrect_tag), corrected_tag)
            # print(f'vlm_cls_input: {vlm_cls_input_dict} incorrect_tag: {incorrect_tag}')
            # print(f'vlm_cls_input[incorrect_tags]: {vlm_cls_input_dict.get(incorrect_tag)} corrected_tags: {corrected_tag}')
        # Print the confusion matrix
        # print("Confusion Matrix with everything:")
        # for predicted_class, corrections in self.confusion_matrix.items():
        #     print(f"{predicted_class}: {dict(corrections)}")

        # Calculate probabilities
        # probabilities = self.calculate_probabilities()
        self.calculate_probabilities()
        # print("\nProbabilities:")
        # for predicted_class, corrections in self.probabilities.items():
        #     for corrected_class, prob in corrections.items():
        #         print(f"P({predicted_class} -> {corrected_class}) = {prob:.2f}")

        #### save all results
        # replace a cls id to a key - landmarks in self.landmark_keys will be deleted
        self.landmark_keys = [vlm_cls_key[vlm_cls_input_idx.index(int(i))] for i in
                              idx_to_remove1]

        self.save_filter_response_json(frame_num, self.output_dir, "results1.json",
                                       self.tag_filter_api.args.system_prompt, filter_txt_input,
                                       str_response1, (items_to_remove1, incorrect_tags, corrected_tags))

        print("remove: ", items_to_remove1)

        if MODIFY_FUNCTION:
            ## TODO: modify landmark class
            modify_dic = self.return_landmarks_to_modify(str_response1, vlm_cls_input, vlm_cls_input_idx)
            print("modify_dic : ", modify_dic)
            self.landmark_keys_to_modify = [vlm_cls_key[vlm_cls_input_idx.index(int(i))] for i in modify_dic.keys()]
            print("landmark_keys_to_modify : ", self.landmark_keys_to_modify)
            self.newclasses_for_landmarks = list(modify_dic.values())  ## need to double check this
            print("newclasses_for_landmarks : ", self.newclasses_for_landmarks)

        return True

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

    def call_remove_landmark_service(self, landmark_key):
        rospy.wait_for_service('remove_landmark')
        try:
            remove_landmark = rospy.ServiceProxy('remove_landmark', RemoveLandmark)
            req = RemoveLandmarkRequest(landmark_key=landmark_key)
            res = remove_landmark(req)
            return res.success
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
            return False

    ## TODO: Modifylandmark service implementation
    # this need to change the lm_to_class, etc
    def call_modify_landmark_service(self, landmark_key, landmark_class):
        rospy.wait_for_service('modify_landmark')
        try:
            modify_landmark = rospy.ServiceProxy('modify_landmark', ModifyLandmark)
            print(f"landmark_key: {landmark_key}, landmark_class: {landmark_class}")
            req = ModifyLandmarkRequest(landmark_key=landmark_key, landmark_class=landmark_class)
            res = modify_landmark(req)
            return res.success
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
            return False


if __name__ == "__main__":
    rospy.init_node("landmarks_comparison_and_removal")
    detector = Compare2DMapAndImage()

    while not rospy.is_shutdown():
        detector.get_model_output()
        rospy.loginfo("Calling service to remove landmark keys: %s" % detector.landmark_keys)
        
        if REMOVE_DUPLICATES:
            print(f'landmark keys to remove due to duplication: {detector.landmark_keys_duplicated}')
            for landmark_key in detector.landmark_keys_duplicated:
                success = detector.call_remove_landmark_service(landmark_key)
                rospy.loginfo("Service call success (duplicated): %s" % success)
        detector.landmark_keys_duplicated = []

        for landmark_key in detector.landmark_keys:
            success = detector.call_remove_landmark_service(landmark_key)
            rospy.loginfo("Service call success: %s" % success)
        detector.landmark_keys = []

        if MODIFY_FUNCTION:
            ## TODO: debug this part
            ## TODO: how to handle the case when detector.landmark_keys overlaps with detector.landmark_keys_to_modify
            for i, landmark_key in enumerate(detector.landmark_keys_to_modify):
                print("here=====================================")
                ## implement dictionary for to have new class name for each landmark_key
                success = detector.call_modify_landmark_service(landmark_key, detector.newclasses_for_landmarks[i])
                rospy.loginfo("Service call success: %s" % success)
            detector.landmark_keys_to_modify = []
            detector.newclasses_for_landmarks = []
        ## change this time if you want to change the frequency of the service call
        rospy.sleep(3)  # Simulate processing time 10
