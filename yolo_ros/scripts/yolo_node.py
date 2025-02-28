#\!/usr/bin/env python3

# Copyright (C) 2023 Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import cv2
import rospy
from typing import List, Dict
from cv_bridge import CvBridge

import torch
from ultralytics import YOLO, YOLOWorld
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks
from ultralytics.engine.results import Keypoints
import os

from std_srvs.srv import SetBool, SetBoolResponse
from sensor_msgs.msg import Image
from yolo_msgs.msg import Point2D
from yolo_msgs.msg import BoundingBox2D
from yolo_msgs.msg import Mask
from yolo_msgs.msg import KeyPoint2D
from yolo_msgs.msg import KeyPoint2DArray
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray
from yolo_msgs.srv import SetClasses, SetClassesResponse


class YoloNode:

    def __init__(self):
        rospy.init_node("yolo_node")

        # params
        self.model_type = rospy.get_param("~model_type", "YOLO")
        self.model = rospy.get_param("~model", "yolov8m.pt")
        self.device = rospy.get_param("~device", "cuda:0")
        self.openvino = rospy.get_param("~openvino", False)

        self.threshold = rospy.get_param("~threshold", 0.5)
        self.iou = rospy.get_param("~iou", 0.5)
        self.imgsz_height = rospy.get_param("~imgsz_height", 640)
        self.imgsz_width = rospy.get_param("~imgsz_width", 640)
        self.half = rospy.get_param("~half", False)
        self.max_det = rospy.get_param("~max_det", 300)
        self.augment = rospy.get_param("~augment", False)
        self.agnostic_nms = rospy.get_param("~agnostic_nms", False)
        self.retina_masks = rospy.get_param("~retina_masks", False)

        self.enable = rospy.get_param("~enable", True)
        
        self.type_to_model = {"YOLO": YOLO, "World": YOLOWorld}
        
        # Set up the CV bridge
        self.cv_bridge = CvBridge()
        
        # Start YOLO model
        try:
            if self.openvino:
                if not os.path.exists(self.model):
                    rospy.logerr(f"Model file '{self.model}' does not exist")
                    return
                
                # Check if model is already an OpenVINO model
                if not self.model.endswith('_openvino_model'):
                    # Get model path without extension
                    model_name = os.path.splitext(self.model)[0]
                    openvino_model_path = f"{model_name}_openvino_model"
                    
                    # Check if converted model already exists
                    if os.path.exists(openvino_model_path):
                        rospy.loginfo(f"Using existing OpenVINO model: {openvino_model_path}")
                        self.model = openvino_model_path
                    else:
                        # Load original model to convert to OpenVINO
                        temp_model = self.type_to_model[self.model_type](self.model)
                        rospy.loginfo(f"Converting {self.model} to OpenVINO format...")
                        openvino_model_path = temp_model.export(format="openvino", dynamic=True)
                        self.model = openvino_model_path
                        del temp_model
                else:
                    rospy.loginfo(f"Using provided OpenVINO model: {self.model}")
                
                # Set device to CPU for OpenVINO
                self.device = "cpu"
                rospy.loginfo("Using CPU for OpenVINO inference")
                
            self.yolo = self.type_to_model[self.model_type](self.model)
            
            if not self.openvino:
                try:
                    rospy.loginfo("Trying to fuse model...")
                    self.yolo.fuse()
                except TypeError as e:
                    rospy.logwarn(f"Error while fuse: {e}")
        except FileNotFoundError:
            rospy.logerr(f"Model file '{self.model}' does not exist")
            return
        except Exception as e:
            rospy.logerr(f"Error loading model: {e}")
            return
        
        # Create publisher
        self._pub = rospy.Publisher("detections", DetectionArray, queue_size=10)
        
        # Create services
        self._enable_srv = rospy.Service("enable", SetBool, self.enable_cb)
        
        if isinstance(self.yolo, YOLOWorld):
            self._set_classes_srv = rospy.Service("set_classes", SetClasses, self.set_classes_cb)
        
        # Create subscriber
        self._sub = rospy.Subscriber("image_raw", Image, self.image_cb, queue_size=1)
        
        rospy.loginfo("YOLO node initialized")
        
        # Register shutdown callback
        rospy.on_shutdown(self.shutdown_cb)

    def shutdown_cb(self):
        rospy.loginfo("Shutting down YOLO node...")
        
        # Clean up YOLO model
        if hasattr(self, 'yolo'):
            del self.yolo
            if "cuda" in self.device:
                rospy.loginfo("Clearing CUDA cache")
                torch.cuda.empty_cache()

    def enable_cb(self, req):
        self.enable = req.data
        res = SetBoolResponse()
        res.success = True
        return res

    def parse_hypothesis(self, results: Results) -> List[Dict]:

        hypothesis_list = []

        if results.boxes:
            box_data: Boxes
            for box_data in results.boxes:
                hypothesis = {
                    "class_id": int(box_data.cls),
                    "class_name": self.yolo.names[int(box_data.cls)],
                    "score": float(box_data.conf),
                }
                hypothesis_list.append(hypothesis)

        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                hypothesis = {
                    "class_id": int(results.obb.cls[i]),
                    "class_name": self.yolo.names[int(results.obb.cls[i])],
                    "score": float(results.obb.conf[i]),
                }
                hypothesis_list.append(hypothesis)

        return hypothesis_list

    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:

        boxes_list = []

        if results.boxes:
            box_data: Boxes
            for box_data in results.boxes:

                msg = BoundingBox2D()

                # get boxes values
                box = box_data.xywh[0]
                msg.center.position.x = float(box[0])
                msg.center.position.y = float(box[1])
                msg.size.x = float(box[2])
                msg.size.y = float(box[3])

                # append msg
                boxes_list.append(msg)

        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                msg = BoundingBox2D()

                # get boxes values
                box = results.obb.xywhr[i]
                msg.center.position.x = float(box[0])
                msg.center.position.y = float(box[1])
                msg.center.theta = float(box[4])
                msg.size.x = float(box[2])
                msg.size.y = float(box[3])

                # append msg
                boxes_list.append(msg)

        return boxes_list

    def parse_masks(self, results: Results) -> List[Mask]:

        masks_list = []

        def create_point2d(x: float, y: float) -> Point2D:
            p = Point2D()
            p.x = x
            p.y = y
            return p

        mask: Masks
        for mask in results.masks:

            msg = Mask()

            msg.data = [
                create_point2d(float(ele[0]), float(ele[1]))
                for ele in mask.xy[0].tolist()
            ]
            msg.height = results.orig_img.shape[0]
            msg.width = results.orig_img.shape[1]

            masks_list.append(msg)

        return masks_list

    def parse_keypoints(self, results: Results) -> List[KeyPoint2DArray]:

        keypoints_list = []

        points: Keypoints
        for points in results.keypoints:

            msg_array = KeyPoint2DArray()

            if points.conf is None:
                continue

            for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):

                if conf >= self.threshold:
                    msg = KeyPoint2D()

                    msg.id = kp_id + 1
                    msg.point.x = float(p[0])
                    msg.point.y = float(p[1])
                    msg.score = float(conf)

                    msg_array.data.append(msg)

            keypoints_list.append(msg_array)

        return keypoints_list

    def image_cb(self, msg):
        if not self.enable:
            return

        # convert image + predict
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = self.yolo.predict(
            source=cv_image,
            verbose=False,
            stream=False,
            conf=self.threshold,
            iou=self.iou,
            imgsz=(self.imgsz_height, self.imgsz_width),
            half=self.half,
            max_det=self.max_det,
            augment=self.augment,
            agnostic_nms=self.agnostic_nms,
            retina_masks=self.retina_masks,
            device=self.device,
        )
        results: Results = results[0].cpu()

        hypothesis = None
        boxes = None
        masks = None
        keypoints = None

        if results.boxes or results.obb:
            hypothesis = self.parse_hypothesis(results)
            boxes = self.parse_boxes(results)

        if results.masks:
            masks = self.parse_masks(results)

        if results.keypoints:
            keypoints = self.parse_keypoints(results)

        # create detection msgs
        detections_msg = DetectionArray()

        for i in range(len(results)):
            aux_msg = Detection()

            if (results.boxes or results.obb) and hypothesis and boxes:
                aux_msg.class_id = hypothesis[i]["class_id"]
                aux_msg.class_name = hypothesis[i]["class_name"]
                aux_msg.score = hypothesis[i]["score"]
                aux_msg.bbox = boxes[i]

            if results.masks and masks:
                aux_msg.mask = masks[i]

            if results.keypoints and keypoints:
                aux_msg.keypoints = keypoints[i]

            detections_msg.detections.append(aux_msg)

        # publish detections
        detections_msg.header = msg.header
        self._pub.publish(detections_msg)

        del results
        del cv_image

    def set_classes_cb(self, req):
        rospy.loginfo(f"Setting classes: {req.classes}")
        self.yolo.set_classes(req.classes)
        rospy.loginfo(f"New classes: {self.yolo.names}")
        return SetClassesResponse()


def main():
    try:
        node = YoloNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
