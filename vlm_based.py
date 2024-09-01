#!/usr/bin/env python3
import os
from dotenv import load_dotenv

load_dotenv()

import pathlib
from typing import Literal

import message_filters
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
import cv2
from semanticslam_ros.msg import ObjectsVector, ObjectVector
from semanticslam_ros.srv import UpdateClasslist, UpdateClasslistResponse, UpdateClasslistRequest
from sensor_msgs.msg import CameraInfo, Image

from PIL import Image as PILImage
from sensor_msgs.msg import Image as RosImage
import sys

np.set_printoptions(threshold=sys.maxsize)

CONF_THRESH = 0.25  # Confidence threshold used for YOLO, default is 0.25
EMBEDDING_LEN = 512  # Length of the embedding vector, default is 512
DETECTOR__CONF_THRESH = 0.6  # 0.76  # Confidence threshold used for the detector, default is 0.5
OBJECT_DEPTH_TRHES = 10.0  # 3.0  # Depth threshold for objects, default is 5.0
DETECTOR_VISUALIZATION_THRESHOLD = 0.6 # visualization threshold for (RAM+DINO)/2

debug = True  # debugging mode
debug_frame_num = 0
debug_output_path = f"{os.environ['DATASETS']}/llm_data/output_ojd"
general_classes_to_remove = ['ceiling', 'floor', 'wall', 'room', 'classroom',
                              'floor', 'ceiling', 'carpet', 'mat',
                             'restroom', 'bathroom', 'dressing',
                             'person', 'child',
                             'black', 'yellow', 'red', 'blue', 'white'] # works only for SLAM landmarks, not for a detector

def unproject(u, v, depth, cam_info):
    """
    Unproject a single pixel to 3D space
    """
    fx = cam_info.K[0]
    fy = cam_info.K[4]
    cx = cam_info.K[2]
    cy = cam_info.K[5]
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    # print("x: ", x)
    # print("y: ", y)
    # print("depth: ", depth)
    # print("fx: ", fx)
    # print("fy: ", fy)
    # print("cx: ", cx)
    # print("cy: ", cy)
    print(f'x: {x:.3f} y: {y:.3f} depth: {depth:.3f}')
    print(f'fx: {fx:.3f} fy: {fy:.3f} cx: {cx:.3f} cy: {cy:.3f}')
    return x, y, depth


def compute_w_h_in_3d(w, h, Z, fx, fy):
    """
    Compute the real world dimensions of a bounding box given its pixel dimensions,
    depth from the camera, and the camera's focal lengths.

    Args:
    - x1, y1: Top-left corner of the bounding box in pixels.
    - x2, y2: Bottom-right corner of the bounding box in pixels.
    - Z: Depth at which the object is located (same unit as desired for output, typically meters).
    - fx, fy: Focal lengths of the camera in pixels (from camera calibration).

    Returns:
    - width, height in real-world units, corresponding to the provided depth.
    """
    width_pixels = w
    height_pixels = h

    width_real = (width_pixels * Z) / fx
    height_real = (height_pixels * Z) / fy

    return width_real, height_real


class ClosedSetDetector:
    """
    This holds an instance of YoloV8 and runs inference
    """

    def __init__(self) -> None:
        assert torch.cuda.is_available()

        method: Literal["yolo", "ram"] = "ram"
        self.method = method
        rospy.loginfo("detector started")
        rospy.loginfo("updateclasslist service started")
        self.update_classes_service = rospy.Service('/update_classlist', UpdateClasslist, self.handle_update_classes)

        if method == "ram":
            import sys
            sys.path.append(os.environ['LLM_MAPPING'])
            sys.path.append(os.environ['DINO'])
            from demos.ram_grounded_sam import load_models, GraundedSamArgs, run_single_image
            # setting up
            GraundedSamArgs.visualize_SAM = False
            self.models = load_models()
            self.inference = run_single_image
            self.classes = {}
            self.classlist =""
            self.classes_from_list = {}

        else:
            from ultralytics import YOLO
            model_file = pathlib.Path(__file__).parent / "../../yolo/yolov8m-seg.pt"
            self.inference = YOLO(model_file)
            self.classes = self.inference.names
            # for idx in self.infernce.names:
            #     self.classes[self.infernce.names[idx]] = idx

        rospy.loginfo("Model loaded")

        self.objs_pub = rospy.Publisher("/camera/objects", ObjectsVector, queue_size=10)
        self.img_pub = rospy.Publisher("/camera/yolo_img", RosImage, queue_size=10)
        self.br = CvBridge()

        # Set up synchronized subscriber 
        # REALSENSE PARAMS
        # cam_info_topic = rospy.get_param("cam_info_topic", "/camera/color/camera_info")
        # rgb_topic = rospy.get_param("rgb_topic", "/camera/color/image_raw")
        # depth_topic = rospy.get_param(
        #     "depth_topic", "/camera/aligned_depth_to_color/image_raw"
        # )

        # Set up synchronized subscriber
        # TUM PARAMS 
        # cam_info_topic = rospy.get_param("cam_info_topic", "/camera/rgb/camera_info")
        # rgb_topic = rospy.get_param("rgb_topic", "/camera/rgb/image_color")
        # depth_topic = rospy.get_param(
        #     "depth_topic", "/camera/depth/image"
        # )

        # JACKAL PARAMS
        cam_info_topic = rospy.get_param("cam_info_topic", "/zed2i/zed_node/rgb/camera_info")
        rgb_topic = rospy.get_param("rgb_topic", "/zed2i/zed_node/rgb/image_rect_color")
        depth_topic = rospy.get_param(
            "depth_topic", "/zed2i/zed_node/depth/depth_registered"
        )
        self.cam_info_sub = message_filters.Subscriber(
            cam_info_topic, CameraInfo, queue_size=1
        )
        self.rgb_img_sub = message_filters.Subscriber(rgb_topic, Image, queue_size=1)
        self.depth_img_sub = message_filters.Subscriber(
            depth_topic, Image, queue_size=1
        )

        # Synchronizer for RGB and depth images
        self.sync = message_filters.ApproximateTimeSynchronizer(
            (self.cam_info_sub, self.rgb_img_sub, self.depth_img_sub), 1, 0.025
        )

        self.sync.registerCallback(self.forward_pass)

    def update_classes_from_list(self):
        """
        Update self.classes dictionary from the string self.classlist.
        """
        # Split the classlist string into a list of class names
        class_names = self.classlist.split(", ")

        # Convert the list of class names into a dictionary with indices as keys
        self.classes = {i: class_name for i, class_name in enumerate(class_names)}
        rospy.loginfo(f"Classes updated: {self.classes}")

    def handle_update_classes(self, req):
        # Example handler, adjust according to your needs
        # Assuming req.data is a string containing the new class list
        self.classlist = req.landmark_class.data
        rospy.loginfo(f"Updated classes: {self.classlist}")

        # Update self.classes after receiving new classlist
        self.update_classes_from_list()
        return UpdateClasslistResponse(success=True)

    def forward_pass(self, cam_info: CameraInfo, rgb: Image, depth: Image) -> None:
        """
        Run YOLOv8 on the image and extract all segmented masks
        """

        objects = ObjectsVector()
        objects.header = rgb.header
        objects.objects = []

        if self.method == "yolo":
            image_cv = self.br.imgmsg_to_cv2(rgb, desired_encoding="bgr8")
            depth_m = self.br.imgmsg_to_cv2(depth, "32FC1")  # Depth in meters
            depth_m = cv2.resize(depth_m, dsize=(1280, 736),
                                 interpolation=cv2.INTER_NEAREST)  # do this for realsense (img dim not a multiple of max stride length 32)

            # Run inference args: https://docs.ultralytics.com/modes/predict/#inference-arguments
            results = self.inference(image_cv, verbose=False, conf=CONF_THRESH, imgsz=(736, 1280))[
                0]  # do this for realsense (img dim not a multiple of max stride length 32)
            if (results.boxes is None) or (results.masks is None):
                return

            # Show the results
            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                im = PILImage.fromarray(im_array[..., ::-1])  # RGB PIL image

                msg_yolo_detections = RosImage()
                msg_yolo_detections.header.stamp = rgb.header.stamp
                msg_yolo_detections.height = im.height
                msg_yolo_detections.width = im.width
                msg_yolo_detections.encoding = "rgb8"
                msg_yolo_detections.is_bigendian = False
                msg_yolo_detections.step = 3 * im.width
                msg_yolo_detections.data = np.array(im).tobytes()
                self.img_pub.publish(msg_yolo_detections)

            masks = results.masks.data.cpu().numpy()
            class_ids = results.boxes.cls.data.cpu().numpy()
            bboxes = results.boxes.xyxy.data.cpu().numpy()
            confs = results.boxes.conf.data.cpu().numpy()

            classNames = [
                "person", "bicycle", "car", "motorcycle", "airplane",
                "bus", "train", "truck", "boat", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "benchhhh", "bird",
                "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                "wine glass", "cupppp", "fork", "knife", "spoon",
                "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut",
                "cake", "chair", "couch", "potted plant", "bed",
                "dining table", "toilet", "tv", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven",
                "toaster", "sink", "refrigerator", "bookkk", "clock",
                "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
            ## once classnames are updated, it can be turned into string
            class_names_string = ", ".join(classNames)

        else:  # self.method = "ram"

            image_cv = self.br.imgmsg_to_cv2(rgb, desired_encoding="bgr8")
            depth_m = self.br.imgmsg_to_cv2(depth, "32FC1")

            height, width = image_cv.shape[:2]
            detections, classes, visualization = self.inference(image=image_cv, models=self.models,
                                                                visualization_threshold=DETECTOR_VISUALIZATION_THRESHOLD)

            if len(classes) < 1:
                return

            # visualization?
            im = PILImage.fromarray(visualization)
            msg_yolo_detections = RosImage()
            msg_yolo_detections.header.stamp = rgb.header.stamp
            msg_yolo_detections.height = im.height
            msg_yolo_detections.width = im.width
            msg_yolo_detections.encoding = "bgr8"
            msg_yolo_detections.is_bigendian = False
            msg_yolo_detections.step = 3 * im.width
            msg_yolo_detections.data = np.array(im).tobytes()

            self.img_pub.publish(msg_yolo_detections)

            #####
            # save frames with the confidences
            if debug:
                global debug_frame_num
                debug_frame_num += 1
                out_name = f"{debug_output_path}/frame_{debug_frame_num:04}.png"
                pathlib.Path(debug_output_path).mkdir(exist_ok=True, parents=True)
                cv2.imwrite(out_name, visualization)

                out_name = f"{debug_output_path}/confidences.txt"
                with open(out_name, 'a', encoding='utf-8') as file:
                    file.write("frame_num\tclass_name\tDINO_conf\t\tRAM_conf\t\th_Dino:h_Ram\t\tDinoXRam\n")
                    file.write(f"detected : {classes}\n")
                    for i in range(len(detections.confidence)):
                        file.write(f"{debug_frame_num}\t{classes[detections.class_id[i]]}\t"
                                   f"{detections.confidence[i]}\t"
                                   f"{detections.ram_conf[i]}\t"
                                   f"{(detections.confidence[i] + detections.ram_conf[i]) / 2}\t"
                                   f"{detections.confidence[i] * detections.ram_conf[i]}\n")
                    file.write("\n")

                #print(f"Data saved to {out_name}")
            #####

            masks, bboxes, class_ids, confs, ram_confs = [], [], [], [], []
            for i, (xyxy, mask, confidence, class_id, _, _) in enumerate(detections):
                # print(f'confidence: {confidence}, conf i: {detections.confidence[i]}, ram conf i: {detections.ram_conf[i]}')
                if class_id is None:
                    continue

                # todo remove this part later
                if classes[class_id] in general_classes_to_remove:  # remove too general classes
                    continue

                bboxes.append(xyxy)

                if mask is None:
                    mask = np.zeros((height, width), dtype=np.float32)
                    mask[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])] = 1.0
                masks.append(mask)

                ## subscribe and update self.classes here?
                if classes[class_id] not in self.classes.values():
                    self.classes[len(self.classes)] = classes[class_id]

                new_cls_id = [key for key, value in self.classes.items() if value == classes[class_id]]

                class_ids.append(new_cls_id[0])
                print(f'class_ids: {class_ids} new_cls_id: {new_cls_id[0]} class_name: {classes[class_id]} self.classes: {self.classes}')
                confs.append(confidence)
                ram_confs.append(detections.ram_conf[i])

            print(list(self.classes.values()))
            class_names_string = ", ".join(list(self.classes.values()))

        if len(masks) == 0:
            return
        for mask, class_id, bboxes, dino_conf, ram_conf in zip(masks, class_ids, bboxes, confs, ram_confs):

            #confidence to get =  (dino's conf + ram's conf) / 2
            conf = (dino_conf + ram_conf) / 2

            # ---- Object Vector ----
            object = ObjectVector()
            class_id = int(class_id)

            # print(f'mask shape: {np.shape(mask)} depth shape: {np.shape(depth_m)}')
            # print(f'class_id: {class_id} object_name:{self.model.names[class_id]} conf: {conf}')
            mask = mask > 0  # Convert to binary 
            obj_depth = np.nanmean(depth_m[mask], dtype=float)
            obj_centroid = np.mean(np.argwhere(mask), axis=0)
            # print(f'obj_depth: {obj_depth} obj_centroid: {obj_centroid}')

            if ((conf < DETECTOR__CONF_THRESH) or (np.isnan(obj_depth)) or (np.isnan(obj_centroid[0]))
                    or (np.isnan(obj_centroid[1])) or (np.isinf(obj_depth)) or (obj_depth > OBJECT_DEPTH_TRHES)):
                print('inf nan passes')
                continue

            print(f'mask shape: {np.shape(mask)} depth shape: {np.shape(depth_m)}')
            print(f'class_id: {class_id} object_name:{self.classes[class_id]} conf: {conf}')
            print(f'obj_depth: {obj_depth} obj_centroid: {obj_centroid}')

            # Unproject centroid to 3D
            x, y, z = unproject(obj_centroid[1], obj_centroid[0], obj_depth, cam_info)
            object.geometric_centroid.x = x
            object.geometric_centroid.y = y
            object.geometric_centroid.z = z

            fx = cam_info.K[0]
            fy = cam_info.K[4]

            object.bbox_width, object.bbox_height = compute_w_h_in_3d(bboxes[2] - bboxes[0], bboxes[3] - bboxes[1],
                                                                      obj_depth, fx, fy)

            if (conf < DETECTOR__CONF_THRESH):
                object.geometric_centroid.x = np.nan
                object.geometric_centroid.y = np.nan
                object.geometric_centroid.z = np.nan

            object.latent_centroid = np.zeros(EMBEDDING_LEN)
            assert class_id < EMBEDDING_LEN, "Class ID > length of vector"

            #####todo: might need to change it?
            object.latent_centroid[class_id] = 1
            object.class_id = class_id
            object.det_conf = conf
            object.ram_conf = ram_conf

            # if ((conf < .9) or (np.isnan(obj_depth)) or (np.isnan(obj_centroid[0])) 
            #     or (np.isnan(obj_centroid[1])) or (np.isinf(obj_depth))):
            #     print(f'inf nan passes')
            #     continue

            objects.objects.append(object)

        ## subscibe class_names_string topic and update the class_names_string? If service was called..

        print(f'classes: {self.classes} classlist: {self.classlist} class_names_string: {class_names_string}')
        ## update the class_names_string
        # if self.classlist != "":
        #     print(f'updating classlist: {self.classlist}')
        #     class_names_string = self.classlist
        #     self.classlist = ""
        
        objects.classlist.data = class_names_string
        self.objs_pub.publish(objects)


if __name__ == "__main__":
    rospy.init_node("closed_set_detector")
    detector = ClosedSetDetector()
    rospy.spin()
