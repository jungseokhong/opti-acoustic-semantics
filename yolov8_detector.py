#!/usr/bin/env python3

import pathlib

import message_filters
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
import cv2
from semanticslam_ros.msg import ObjectsVector, ObjectVector
from sensor_msgs.msg import CameraInfo, Image
from ultralytics import YOLO
from PIL import Image as PILImage
from sensor_msgs.msg import Image as RosImage
import sys

np.set_printoptions(threshold=sys.maxsize)

## CONF_THRESH = 0.7 causes discontinuity in the trajectory
CONF_THRESH = 0.25  # Confidence threshold used for YOLO, default is 0.25
EMBEDDING_LEN = 512  # Length of the embedding vector, default is 512
DETECTOR__CONF_THRESH = 0.7  # Confidence threshold used for the detector, default is 0.5
OBJECT_DEPTH_TRHES = 3.0  # Depth threshold for objects, default is 5.0

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
        model_file = pathlib.Path(__file__).parent / "../../yolo/yolov8m-seg.pt"
        self.model = YOLO(model_file)
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

    def forward_pass(self, cam_info: CameraInfo, rgb: Image, depth: Image) -> None:
        """
        Run YOLOv8 on the image and extract all segmented masks
        """

        objects = ObjectsVector()
        # objects.header.stamp = rospy.Time.now() # rgb.header ## [TODO]: check if I can use RGB header timestamp for everything
        objects.header = rgb.header ## [TODO]: check if I can use RGB header timestamp for everything
        objects.objects = []

        image_cv = self.br.imgmsg_to_cv2(rgb, desired_encoding="bgr8")
        # depth_m = (
        #     self.br.imgmsg_to_cv2(depth, desired_encoding="passthrough") / 1000.0 # for TUM depth is in meters already, for realsense it is in mm
        # )  # Depth in meters
        depth_m = self.br.imgmsg_to_cv2(depth, "32FC1") # Depth in meters
        # print(f'+++++ depth mean: {np.nanmean(depth_m)} depth max: {np.nanmax(depth_m)} depth min: {np.nanmin(depth_m)} +++++')
        # resizing depth is probably not needed
        depth_m = cv2.resize(depth_m, dsize=(1280, 736), interpolation=cv2.INTER_NEAREST) # do this for realsense (img dim not a multiple of max stride length 32)
        # print(f'===== depth mean: {np.nanmean(depth_m)} depth max: {np.nanmax(depth_m)} depth min: {np.nanmin(depth_m)} =====')
        # depth_m = cv2.resize(depth_m, dsize=(640, 360), interpolation=cv2.INTER_NEAREST) # do this for zed (img dim not a multiple of max stride length 32)


        # Run inference args: https://docs.ultralytics.com/modes/predict/#inference-arguments
        # [TODO]: this causes error in the later stage. need to put the results back to the original image size
        results = self.model(image_cv, verbose=False, conf=CONF_THRESH, imgsz=(736, 1280))[0] # do this for realsense (img dim not a multiple of max stride length 32)
        # results = self.model(image_cv, verbose=False, conf=CONF_THRESH, imgsz=(360, 640))[0] # do this for realsense (img dim not a multiple of max stride length 32)
        #results = self.model(image_cv, verbose=False, conf=CONF_THRESH)[0]

        # Extract segmentation masks
        if (results.boxes is None) or (results.masks is None):
            return

        # Show the results
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            print("im_array shape: ", im_array.shape)
            im = PILImage.fromarray(im_array[..., ::-1])  # RGB PIL image
            
            msg_yolo_detections = RosImage()
            msg_yolo_detections.header.stamp = rgb.header.stamp ## original code. Is this necessary?
            # msg_yolo_detections.header.stamp = rospy.Time.now() ## time step updated for syncing with map info
            msg_yolo_detections.height = im.height
            msg_yolo_detections.width = im.width
            msg_yolo_detections.encoding = "rgb8"
            msg_yolo_detections.is_bigendian = False
            msg_yolo_detections.step = 3 * im.width
            msg_yolo_detections.data = np.array(im).tobytes()
            self.img_pub.publish(msg_yolo_detections)
            # im.show()  # show image
            # im.save('results.jpg')  # save image

        masks = results.masks.data.cpu().numpy()
        class_ids = results.boxes.cls.data.cpu().numpy()
        bboxes = results.boxes.xywh.data.cpu().numpy() # boxes.xyxy or boxes.xywh   # box with xywh format, (N, 4)
        confs = results.boxes.conf.data.cpu().numpy()


        classNames = [
                "person", "bicycle", "car", "motorcycle", "airplane",
                "bus", "train", "truck", "boat", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "bench", "bird",
                "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                "wine glass", "cup", "fork", "knife", "spoon",
                "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut",
                "cake", "chair", "couch", "potted plant", "bed",
                "dining table", "toilet", "tv", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven",
                "toaster", "sink", "refrigerator", "book", "clock",
                "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        ## once classnames are updated, it can be turned into string
        class_names_string = ", ".join(classNames)
        # print(type(class_names_string), class_names_string)


        if len(masks) == 0:
            return
        for mask, class_id, bbox, conf in zip(masks, class_ids, bboxes, confs):
            # ---- Object Vector ----
            object = ObjectVector()
            class_id = int(class_id)

            # print(f'mask shape: {np.shape(mask)} depth shape: {np.shape(depth_m)}')
            # print(f'class_id: {class_id} object_name:{self.model.names[class_id]} conf: {conf}')
            mask = mask > 0  # Convert to binary 
            obj_depth = np.nanmean(depth_m[mask], dtype=float)             
            obj_centroid = np.mean(np.argwhere(mask), axis=0)
            ## rescale obj_centroid to the original image size

            #[TODO] don't hardcode 540 960 image sizes
            obj_centroid[0] = obj_centroid[0] * 540/736
            obj_centroid[1] = obj_centroid[1] * 960/1280
            # print(f'obj_depth: {obj_depth} obj_centroid: {obj_centroid}')

            if ((conf < DETECTOR__CONF_THRESH) or (np.isnan(obj_depth)) or (np.isnan(obj_centroid[0])) 
                or (np.isnan(obj_centroid[1])) or (np.isinf(obj_depth)) or (obj_depth > OBJECT_DEPTH_TRHES)):
                print(f'inf nan passes')
                continue

            print(f'mask shape: {np.shape(mask)} depth shape: {np.shape(depth_m)}')
            print(f'class_id: {class_id} object_name:{self.model.names[class_id]} conf: {conf}')
            print(f'obj_depth: {obj_depth} obj_centroid: {obj_centroid}')
            print(f'bbox: {bbox}')



            # Unproject centroid to 3D in camera frame // addcentroidsrealclass projects the centroid to 3D in body frame
            x, y, z = unproject(obj_centroid[1], obj_centroid[0], obj_depth, cam_info)
            object.geometric_centroid.x = x
            object.geometric_centroid.y = y
            object.geometric_centroid.z = z
            
            if (conf < DETECTOR__CONF_THRESH):
                object.geometric_centroid.x = np.nan
                object.geometric_centroid.y = np.nan
                object.geometric_centroid.z = np.nan

            object.latent_centroid = np.zeros(EMBEDDING_LEN)
            assert class_id < EMBEDDING_LEN, "Class ID > length of vector"
            object.latent_centroid[class_id] = 1
            object.class_id = class_id

            fx = cam_info.K[0]
            fy = cam_info.K[4]

            object.bbox_width, object.bbox_height = compute_w_h_in_3d(bbox[2], bbox[3], obj_depth, fx, fy)

            # object.bbox_height = 0
            # object.bbox_width = 0

            # if ((conf < .9) or (np.isnan(obj_depth)) or (np.isnan(obj_centroid[0])) 
            #     or (np.isnan(obj_centroid[1])) or (np.isinf(obj_depth))):
            #     print(f'inf nan passes')
            #     continue


            objects.objects.append(object)
            
        objects.classlist.data = class_names_string
        self.objs_pub.publish(objects)


if __name__ == "__main__":
    rospy.init_node("closed_set_detector")
    detector = ClosedSetDetector()
    rospy.spin()