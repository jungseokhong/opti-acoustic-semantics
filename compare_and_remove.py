#!/usr/bin/env python3
from dotenv import load_dotenv
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
from semanticslam_ros.msg import MapInfo, ObjectsVector

from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as RosImage
from scipy.spatial.transform import Rotation as R
from semanticslam_ros.srv import RemoveClass, RemoveClassRequest
from semanticslam_ros.srv import RemoveLandmark, RemoveLandmarkRequest
from ast import literal_eval

from vlm_filter_utils import vision_filter

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


class Compare2DMapAndImage:
    def __init__(self):

        self.save_projections = True
        self.output_dir = Path(os.environ['DATASETS']) / "llm_data/rosbag_output_bbox"
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

        #####comparison set
        self.compare_promts = False
        if self.compare_promts:
            from vlm_filter_utils import vision_another_filter
            self.vlm_filter_com = vision_agent(vision_another_filter)
            self.output_com_dir = Path(os.environ['DATASETS']) / "llm_data/rosbag_output_bbox_comp"

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
        position, orientation, landmark_points, landmark_classes, landmark_widths, landmark_heights, landmark_keys = self.parse_data(
            map_info)
        # Project landmarks to the image
        projected_image = self.projectLandmarksToImage(position, orientation, landmark_points, landmark_classes,
                                                       landmark_widths, landmark_heights, landmark_keys, img=yoloimg_cv)

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

    def tag_edge(self, edge, tlx, tly, brx, bry, box_size, add_w):
        if edge == "tl":  # top_left
            if tly - box_size < 0:
                tlp_y = tly  # inside
                brp_y = tly + box_size
            else:
                tlp_y = tly - box_size
                brp_y = tly
            tlp_x = tlx
            brp_x = tlx + box_size + add_w
        elif edge == "tr":
            if tly - box_size < 0:
                tlp_y = tly  # inside
                brp_y = tly + box_size
            else:
                tlp_y = tly - box_size
                brp_y = tly
            tlp_x = brx - box_size - add_w
            brp_x = brx
        elif edge == "lt":
            if tlx - box_size - add_w < 0:
                tlp_x = tlx  # inside
                brp_x = tlx + box_size + add_w
            else:
                tlp_x = tlx - box_size - add_w
                brp_x = tlx
            tlp_y = tly
            brp_y = tly + box_size
        elif edge == "rt":
            if brx + box_size + add_w > self.img_width:
                tlp_x = brx - box_size - add_w  # inside
                brp_x = brx
            else:
                tlp_x = brx
                brp_x = brx + box_size + add_w
            tlp_y = tly
            brp_y = tly + box_size
        elif edge == "bl":
            if bry + box_size > self.img_height:
                tlp_y = bry - box_size  # inside
                brp_y = bry
            else:
                tlp_y = bry
                brp_y = bry + box_size
            tlp_x = tlx
            brp_x = tlx + box_size + add_w
        else:  # edge == "br"
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
        add_w = 10
        edge = ["tl", "tr", "bl", "br", "lt", "rt"]

        for edge_mode in edge:
            (tlp_x, tlp_y), (brp_x, brp_y) = self.tag_edge(edge_mode, tlx, tly, brx, bry, box_size, add_w)
            if tag_area == []:
                #print(edge_mode)
                tag_area.append((tlp_x, tlp_y, brp_x, brp_y))
                return ((tlp_x, tlp_y), (brp_x, brp_y))
            overlapping = False
            for area in tag_area:
                if (area[0] <= brp_x  and  area[2] >= tlp_x) and (area[1] <= brp_y and area[3] >= tlp_y):
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
        for i, (point_3d, landmark_key, point_2d) in enumerate(zip(landmark_points, landmark_keys, points_2d_homo.T)):
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
                tlx = x - scaled_width // 2 if x - scaled_width // 2 >= 0 else 0
                tly = y - scaled_height // 2 if y - scaled_height // 2 >= 0 else 0
                brx = x + scaled_width // 2 if x + scaled_width // 2 < self.img_width else self.img_width - 1
                bry = y + scaled_height // 2 if y + scaled_height // 2 < self.img_height else self.img_height - 1

                # tag_box
                b_size = 20
                tag_tl, tag_br = self.tag_box(tlx, tly, brx, bry, b_size, tag_area)

                # Draw the bounding box
                pre_projected = projected_image.copy()
                cv2.rectangle(projected_image, tag_tl, tag_br, color, -1)  # tag
                cv2.rectangle(projected_image, (tlx, tly), (brx, bry), color, 2)  # boudingbox

                alpha = 0.4
                projected_image = cv2.addWeighted(projected_image, alpha, pre_projected, 1 - alpha, 0, pre_projected)

                #tag_name = f"[{i}]"
                #cv2.putText(projected_image, tag_name, (tag_tl[0], tag_br[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                #            (0, 0, 0), 1)
                #cv2.rectangle(projected_image, (tlx, tly), (brx, bry), color, 1)

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
            cv2.putText(projected_image, tag_name, (tag[0], tag[3] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 1)


        self.vlm_cls_key = [np.int64(d["landmark_key"]) for d in obj]  # key
        self.vlm_cls_input = [d["label"] for d in obj]  # class name
        self.vlm_cls_input_idx = [d["i"] for d in obj]  # index

        if self.save_projections:
            self.frame_num += 1
            # json_out["image_idx"] = "{:05d}_ori.png".format(self.frame_num)
            json_out["{:05d}.png".format(self.frame_num)] = {"contents": obj}
            # json_out["contents"] = obj

            self.save_img(img, projected_image)
            self.save_json(json_out, self.output_dir)

        self.vlm_img_input = projected_image
        return projected_image

    def save_img(self, img, projected_image):

        output_path = self.output_dir  # / time_string
        output_path.mkdir(parents=True, exist_ok=True)

        _output_path = output_path / "{:05d}.png".format(self.frame_num)
        cv2.imwrite(str(_output_path), img)

        _output_path = output_path / "{:05d}_proj.png".format(self.frame_num)
        cv2.imwrite(str(_output_path), projected_image)

    def save_json(self, json_out, output_dir):
        # name = json_out["image_idx"][:-4] + ".json"
        json_path = output_dir / "results.json"

        if json_path.exists():
            f = open(json_path)
            data = json.load(f)
        else:
            data = {}

        for key in json_out.keys():
            data[key] = json_out[key]

        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

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

    def call_api(self, vlm_filter, vlm_img_input, txt_input):
        vlm_filter.reset_memory()
        vlm_response = vlm_filter.call_vision_agent_with_image_input(vlm_img_input, txt_input, self.client)
        str_response = return_str(vlm_response)
        return str_response

    def return_landmarks_to_remove(self, str_response, vlm_cls_input, vlm_cls_input_idx):
        ## Extract the part of the string that represents the list
        list_from_idx = str_response.split('=')[-1].strip()

        try:
            list_from_idx = literal_eval(list_from_idx)
        except:
            print(f"\nerror during extract the result list..{list_from_idx}\n")
            list_from_idx = []  # prevent the program from stopping

        # replace a cls id to a cls name
        list_from_string = [vlm_cls_input[vlm_cls_input_idx.index(int(i))] for i in
                            list_from_idx]

        return list_from_string, list_from_idx

    def save_response_json(self, prompts, txt_input, str_response, frame_num, list_from_string, output_dir):
        ##save it to json file
        json_path = output_dir / "results.json"
        if json_path.exists():
            f = open(json_path)
            data = json.load(f)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            data = {"prompts": prompts.split('\n'), "{:05d}.png".format(frame_num): {}}

        if "prompts" not in data:
            new_data = {}
            new_data["prompts"] = prompts.split('\n')
            for key in data.keys():
                new_data[key] = data[key]
            data = new_data

        if "{:05d}.png".format(frame_num) not in data:
            data["{:05d}.png".format(frame_num)] = {}

        str_response = str_response.split('\n')
        json_out = {"text_input": txt_input, "vlm_response": str_response, "filtered out": list_from_string}
        data["{:05d}.png".format(frame_num)]["vlm_filter"] = json_out

        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

    def get_model_output(self):

        if self.frame_num < 1:
            return True

        if len(self.vlm_cls_input) < 1:
            print("no landmark")
            return True

        # if self.frame_num not in [1,20,24,36,53,64,79,99,117,132,146,163,177,178]:
        #     return True

        # copy necessary vars
        frame_num = self.frame_num
        vlm_cls_input_idx = self.vlm_cls_input_idx
        vlm_cls_input = self.vlm_cls_input
        vlm_cls_key = self.vlm_cls_key
        vlm_img_input = self.vlm_img_input.copy()

        print("frame : ", frame_num)
        print("tags :", vlm_cls_input, vlm_cls_input_idx)
        self.vlm_cls_input = []  # in order to prevent calling vlm repeatedly with the same input

        ##edit a text input
        txt_input = []
        for cls_name, cls_num in zip(vlm_cls_input, vlm_cls_input_idx):
            txt_input.append(f"{cls_name}: {cls_num}")
        txt_input = ", ".join(txt_input)
        print(f"[text input] {txt_input}")

        # save an input img
        # cv2.imwrite(str(self.output_dir / "{:05d}_input.png".format(frame_num)), vlm_img_input)

        if not self.compare_promts:
            self.vlm_filter.reset_memory()
            str_response = self.call_api(self.vlm_filter, vlm_img_input, txt_input)
            items_to_remove, idx_to_remove = self.return_landmarks_to_remove(str_response, vlm_cls_input,
                                                                             vlm_cls_input_idx)

            # replace a cls id to a key
            self.landmark_keys = [vlm_cls_key[vlm_cls_input_idx.index(int(i))] for i in
                                  idx_to_remove]

            self.save_response_json(self.vlm_filter.args.system_prompt, txt_input, str_response, frame_num,
                                    items_to_remove, self.output_dir)
            print("remove - experiment : ", items_to_remove)
        else:  # multi-threading
            ###
            from concurrent.futures import ThreadPoolExecutor
            prompt = """
cup: 0, book: 1, baseball hat: 3, baseball hat: 4, hat: 7

Examples of each step's output for the given image and its tags:
Step 1
    - Tag 0 (cup): Incorrect.The bounding box contains a ball.
    - Tag 1 (book): Correct. The bounding box contains a book.
    - Tag 3 (baseball hat): Correct. The bounding box contains a baseball hat.
    - Tag 4 (baseball hat): Correct. The bounding box contains a baseball hat but it's duplicated.  
    - Tag 7 (hat): Correct. The bounding box contains a hat.
            
Step 2
    - Tag 3,4, and 7 are pointing to one object. Baseball hat is more precise tag than hat since there is LA mark on it. Considering the spatial relationships between the remaining tags, Tag 3 Focuses on a smaller part of the baseball hat, not covering the entire object. Tag 4 The most precise one.
            
Step 3
    - unmatched_tags = [0]
    - unmatched_tags = [3, 7]
    
Step 4
    - unmatched_tags = [0, 3, 7]
"""
            user_commant = """cup: 0, book: 1, baseball hat: 3, baseball hat: 4, hat: 7"""
            assistant_commant= """Step 1
    - Tag 0 (cup): Incorrect.The bounding box contains a ball.
    - Tag 1 (book): Correct. The bounding box contains a book.
    - Tag 3 (baseball hat): Correct. The bounding box contains a baseball hat.
    - Tag 4 (baseball hat): Correct. The bounding box contains a baseball hat but it's duplicated.  
    - Tag 7 (hat): Correct. The bounding box contains a hat.
            
Step 2
    - Tag 3,4, and 7 are pointing to one object. Baseball hat is more precise tag than hat since there is LA mark on it. Considering the spatial relationships between the remaining tags, Tag 3 Focuses on a smaller part of the baseball hat, not covering the entire object. Tag 4 The most precise one.
            
Step 3
    - unmatched_tags = [0]
    - unmatched_tags = [3, 7]
    
Step 4
    - unmatched_tags = [0, 3, 7]"""

            self.vlm_filter.reset_memory()
            self.vlm_filter.reset_with_img(role = "system", prompt=prompt, img = "/home/beantown/ran/llm_ws/src/maxmixtures/opti-acoustic-semantics/example_image.png")
            ###

            self.vlm_filter_com.reset_memory()
            self.vlm_filter_com.reset_with_img(role="user", prompt=user_commant,
                                           img="/home/beantown/ran/llm_ws/src/maxmixtures/opti-acoustic-semantics/example_image.png")
            self.vlm_filter_com.add_memory_with_prompts(role = "assistant", prompt = assistant_commant)

            cond1 = {"vlm_filter": self.vlm_filter, "vlm_img_input": vlm_img_input, "txt_input": txt_input}
            cond2 = {"vlm_filter": self.vlm_filter_com, "vlm_img_input": vlm_img_input, "txt_input": txt_input}
            with ThreadPoolExecutor() as executor:
                experi = executor.submit(self.call_api, **cond1)
                compa = executor.submit(self.call_api, **cond2)
                exp_response = experi.result()
                comp_response = compa.result()

            items_to_remove_exp, idx_to_remove_exp = self.return_landmarks_to_remove(exp_response, vlm_cls_input,
                                                                                     vlm_cls_input_idx)
            items_to_remove_com, _ = self.return_landmarks_to_remove(comp_response, vlm_cls_input, vlm_cls_input_idx)
            # replace a cls id to a key
            self.landmark_keys = [vlm_cls_key[vlm_cls_input_idx.index(int(i))] for i in
                                  idx_to_remove_exp]

            self.save_response_json(self.vlm_filter.args.system_prompt, txt_input, exp_response, frame_num,
                                    items_to_remove_exp, self.output_dir)
            self.save_response_json(self.vlm_filter_com.args.system_prompt, txt_input, comp_response, frame_num,
                                    items_to_remove_com, self.output_com_dir)

            print("remove - experiment : ", items_to_remove_exp)
            print("remove - comparison : ", items_to_remove_com)

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


if __name__ == "__main__":
    rospy.init_node("landmarks_comparison_and_removal")
    detector = Compare2DMapAndImage()

    while not rospy.is_shutdown():
        detector.get_model_output()

        rospy.loginfo("Calling service to remove landmark keys: %s" % detector.landmark_keys)
        for landmark_key in detector.landmark_keys:
            success = detector.call_remove_landmark_service(landmark_key)
            rospy.loginfo("Service call success: %s" % success)
        detector.landmark_keys = []
        ## change this time if you want to change the frequency of the service call
        rospy.sleep(3)  # Simulate processing time 10
