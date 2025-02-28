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


import rospy

import cv2
import numpy as np
import message_filters
from cv_bridge import CvBridge

from ultralytics.engine.results import Boxes
from ultralytics.trackers.basetrack import BaseTrack
from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml

from sensor_msgs.msg import Image
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray


class TrackingNode:

    def __init__(self) -> None:
        rospy.init_node("tracking_node")

        # params
        self.tracker_name = rospy.get_param("~tracker", "bytetrack.yaml")

        self.cv_bridge = CvBridge()

    def configure(self):
        rospy.loginfo("Configuring...")

        self.tracker = self.create_tracker(self.tracker_name)
        self._pub = rospy.Publisher("tracking", DetectionArray, queue_size=10)

        rospy.loginfo("Configured")

    def activate(self):
        rospy.loginfo("Activating...")

        # subs
        self.image_sub = message_filters.Subscriber("image_raw", Image)
        self.detections_sub = message_filters.Subscriber("detections", DetectionArray)

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.image_sub, self.detections_sub), 10, 0.5
        )
        self._synchronizer.registerCallback(self.detections_cb)

        rospy.loginfo("Activated")

    def deactivate(self):
        rospy.loginfo("Deactivating...")

        self.image_sub.unregister()
        self.detections_sub.unregister()

        del self._synchronizer
        self._synchronizer = None

        rospy.loginfo("Deactivated")

    def cleanup(self):
        rospy.loginfo("Cleaning up...")

        del self.tracker
        self._pub.unregister()

        rospy.loginfo("Cleaned up")

    def shutdown(self):
        rospy.loginfo("Shutting down...")
        self.cleanup()
        rospy.loginfo("Shutted down")

    def create_tracker(self, tracker_yaml: str) -> BaseTrack:

        TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
        check_requirements("lap")  # for linear_assignment

        tracker = check_yaml(tracker_yaml)
        cfg = IterableSimpleNamespace(**yaml_load(tracker))

        assert cfg.tracker_type in [
            "bytetrack",
            "botsort",
        ], f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=1)
        return tracker

    def detections_cb(self, img_msg: Image, detections_msg: DetectionArray) -> None:

        tracked_detections_msg = DetectionArray()
        tracked_detections_msg.header = img_msg.header

        # convert image
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # parse detections
        detection_list = []
        detection: Detection
        for detection in detections_msg.detections:

            detection_list.append(
                [
                    detection.bbox.center.position.x - detection.bbox.size.x / 2,
                    detection.bbox.center.position.y - detection.bbox.size.y / 2,
                    detection.bbox.center.position.x + detection.bbox.size.x / 2,
                    detection.bbox.center.position.y + detection.bbox.size.y / 2,
                    detection.score,
                    detection.class_id,
                ]
            )

        # tracking
        if len(detection_list) > 0:

            det = Boxes(np.array(detection_list), (img_msg.height, img_msg.width))
            tracks = self.tracker.update(det, cv_image)

            if len(tracks) > 0:

                for t in tracks:

                    tracked_box = Boxes(t[:-1], (img_msg.height, img_msg.width))
                    tracked_detection: Detection = detections_msg.detections[int(t[-1])]

                    # get boxes values
                    box = tracked_box.xywh[0]
                    tracked_detection.bbox.center.position.x = float(box[0])
                    tracked_detection.bbox.center.position.y = float(box[1])
                    tracked_detection.bbox.size.x = float(box[2])
                    tracked_detection.bbox.size.y = float(box[3])

                    # get track id
                    track_id = ""
                    if tracked_box.is_track:
                        track_id = str(int(tracked_box.id))
                    tracked_detection.id = track_id

                    # append msg
                    tracked_detections_msg.detections.append(tracked_detection)

        # publish detections
        self._pub.publish(tracked_detections_msg)


def main():
    try:
        node = TrackingNode()
        node.configure()
        node.activate()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
