# #!/usr/bin/env python3

# ###########################################################################################################
# import time
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from ultralytics import YOLO
# import ros2_numpy as rnp

# from std_msgs.msg import String, Float32MultiArray




# class UltralyticsNode(Node):
#     def __init__(self):
#         super().__init__('ultralytics')

#         # Load YOLO models for detection and segmentation
#         self.detection_model = YOLO("yolov8m.pt")
#         self.segmentation_model = YOLO("yolov8m-seg.pt")

#         # Create publishers for the annotated images
#         self.det_image_pub = self.create_publisher(Image, "/ultralytics/detection/image", 5)
#         self.seg_image_pub = self.create_publisher(Image, "/ultralytics/segmentation/image", 5)
#         self.classes_pub = self.create_publisher(String, "/ultralytics/classes", 5)
#         self.bbox_pub = self.create_publisher(Float32MultiArray, "/ultralytics/detection/bounding_boxes", 5)

#         # Create a subscriber to the camera image topic
#         self.subscription = self.create_subscription(
#             Image,
#             '/zed/zed_node/left_raw/image_raw_color',  # Adjust this topic name if necessary
#             self.callback,
#             10)

#     def callback(self, data):
#         # print("Received image")
#         """Callback function to process image and publish annotated images."""
#         # Convert the incoming ROS Image message to a NumPy array using ros2_numpy
#         print(type(data))
#         array = rnp.numpify(data)
#         # print(array.shape)
#         # Determine the encoding and adjust the array if necessary
#         if data.encoding in ['rgba8', 'bgra8']:  # Check if the image has an alpha channel
#             array = array[:, :, :3]  # Remove the alpha channel
#         # print(array.shape)
#         # Check if there are any subscribers to the detection image topic
#         if self.det_image_pub.get_subscription_count() > 0:
#             # Run the detection model on the NumPy array (image)
#             det_result = self.detection_model(array)
#             # print(det_result)

#             # Plot the detection results on the image and return it as a NumPy array
#             det_annotated = det_result[0].plot(show=False)
#             # Convert the annotated NumPy array back to a ROS Image message and publish it
#             det_image_msg = rnp.msgify(Image, det_annotated, encoding="rgb8")
#             # print(det_image_msg)
#             self.det_image_pub.publish(det_image_msg)

#             # Prepare and publish bounding box data
#             if self.bbox_pub.get_subscription_count() > 0:
#                 bounding_boxes = Float32MultiArray()
#                 for box in det_result[0].boxes:
#                     # Extract bounding box data (x_min, y_min, x_max, y_max)
#                     x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
#                     class_id = int(box.cls.cpu().numpy())  # Class ID
#                     confidence = float(box.conf.cpu().numpy())  # Confidence score

#                     # Add data to the bounding_boxes array
#                     bounding_boxes.data.extend([x_min, y_min, x_max, y_max, class_id, confidence])

#                 self.bbox_pub.publish(bounding_boxes)
#         if self.seg_image_pub.get_subscription_count() > 0:
#             # Run the segmentation model on the NumPy array (image)
#             seg_result = self.segmentation_model(array)
#             # Plot the segmentation results on the image and return it as a NumPy array
#             seg_annotated = seg_result[0].plot(show=False)
#             # Convert the annotated NumPy array back to a ROS Image message and publish it
#             seg_image_msg = rnp.msgify(Image, seg_annotated, encoding="rgb8")
#             self.seg_image_pub.publish(seg_image_msg)

#         if self.classes_pub.get_subscription_count() > 0:
#             det_result = self.detection_model(array)
#             classes = det_result[0].boxes.cls.cpu().numpy().astype(int)
#             names = [det_result[0].names[i] for i in classes]
#             self.classes_pub.publish(String(data=str(names)))

# def main(args=None):
#     rclpy.init(args=args)  # Initialize the rclpy library
#     node = UltralyticsNode()  # Create the node

#     try:
#         rclpy.spin(node)  # Keep the node running and processing callbacks
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()  # Destroy the node on shutdown
#         rclpy.shutdown()  # Shutdown the rclpy library

# if __name__ == '__main__':
#     main()



#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

from std_msgs.msg import String, Float32MultiArray
print("Hello world")

class UltralyticsNode(Node):
    def __init__(self):
        super().__init__('ultralytics')

        # Load YOLO models for detection and segmentation
        self.detection_model = YOLO("yolov8m.pt")
        self.segmentation_model = YOLO("yolov8m-seg.pt")

        # Create publishers for the annotated images
        self.det_image_pub = self.create_publisher(Image, "/ultralytics/detection/image", 5)
        self.seg_image_pub = self.create_publisher(Image, "/ultralytics/segmentation/image", 5)
        self.classes_pub = self.create_publisher(String, "/ultralytics/classes", 5)
        self.bbox_pub = self.create_publisher(Float32MultiArray, "/ultralytics/detection/bounding_boxes", 5)

        # Create a subscriber to the camera image topic
        self.subscription = self.create_subscription(
            Image,
            '/zed/zed_node/left_raw/image_raw_color',
            self.callback,
            10)

        # Create CvBridge to convert ROS images to OpenCV format
        self.bridge = CvBridge()

    def callback(self, data):
        try:
            # Convert ROS Image message to a NumPy array using CvBridge
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            if data.encoding in ['rgba8', 'bgra8']:  # Check if the image has an alpha channel
                cv_image = cv_image[:, :, :3]  # Remove the alpha channel

            # Detection and Segmentation processing
            self.process_detection(cv_image)
            self.process_segmentation(cv_image)
            self.publish_classes(cv_image)

        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert image: {str(e)}')

    def process_detection(self, image):
        if self.det_image_pub.get_subscription_count() > 0:
            det_result = self.detection_model(image)
            det_annotated = det_result[0].plot(show=False)
            det_image_msg = self.bridge.cv2_to_imgmsg(det_annotated, encoding="rgb8")
            self.det_image_pub.publish(det_image_msg)
            self.publish_bounding_boxes(det_result)

    def process_segmentation(self, image):
        if self.seg_image_pub.get_subscription_count() > 0:
            seg_result = self.segmentation_model(image)
            seg_annotated = seg_result[0].plot(show=False)
            seg_image_msg = self.bridge.cv2_to_imgmsg(seg_annotated, encoding="rgb8")
            self.seg_image_pub.publish(seg_image_msg)

    def publish_classes(self, image):
        if self.classes_pub.get_subscription_count() > 0:
            det_result = self.detection_model(image)
            classes = det_result[0].boxes.cls.cpu().numpy().astype(int)
            names = [det_result[0].names[i] for i in classes]
            self.classes_pub.publish(String(data=str(names)))

    def publish_bounding_boxes(self, det_result):
        if self.bbox_pub.get_subscription_count() > 0:
            bounding_boxes = Float32MultiArray()
            for box in det_result[0].boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls.cpu().numpy())
                confidence = float(box.conf.cpu().numpy())
                bounding_boxes.data.extend([x_min, y_min, x_max, y_max, class_id, confidence])
            self.bbox_pub.publish(bounding_boxes)

def main(args=None):
    rclpy.init(args=args)
    node = UltralyticsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
