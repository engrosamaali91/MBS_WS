# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from std_msgs.msg import String
# from cv_bridge import CvBridge
# import torch
# import cv2
# import numpy as np
# from ultralytics import YOLO

# class ZedYoloV8Node(Node):
#     def __init__(self):
#         super().__init__('zed_yolov8_node')
#         self.bridge = CvBridge()

#         # Publisher for bounding boxes
#         self.bb_publisher = self.create_publisher(String, 'bounding_boxes', 10)
        
#         # YOLOv8 model initialization
#         self.model = YOLO('yolov8m.pt')

#         # Subscribe to the ZED camera image topic
#         self.subscription = self.create_subscription(
#             Image,
#             '/zed/zed_node/rgb/image_rect_color',  # Update this to your ZED image topic
#             self.image_callback,
#             10
#         )
#         self.subscription  # Prevent unused variable warning

#     def image_callback(self, msg):
#         # Convert ROS Image message to OpenCV image
#         cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
#         # Perform YOLOv8 inference
#         results = self.model(cv_image)
        
#         # Extract detected bounding boxes
#         detections = self.detections_to_custom_box(results)
        
#         # Publish the bounding boxes
#         detection_msg = self.format_detections(detections)
#         self.bb_publisher.publish(String(data=detection_msg))

#     def detections_to_custom_box(self, results):
#         output = []
#         for result in results:
#             for bbox in result.boxes:
#                 obj = {
#                     "label": int(bbox.cls.item()),  # Convert tensor to int
#                     "confidence": float(bbox.conf.item()),  # Convert tensor to float
#                     "bbox": bbox.xywh.tolist()  # Convert tensor to list
#                 }
#                 output.append(obj)
#         return output

#     def format_detections(self, detections):
#         output = []
#         for det in detections:
#             bbox = f"Class: {det['label']}, Confidence: {det['confidence']:.2f}, BBox: {det['bbox']}"
#             output.append(bbox)
#         return '\n'.join(output)

# def main(args=None):
#     rclpy.init(args=args)
#     node = ZedYoloV8Node()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()




# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from std_msgs.msg import Float32MultiArray
# from cv_bridge import CvBridge
# import cv2
# import numpy as np
# from ultralytics import YOLO

# class ZedYoloV8Node(Node):
#     def __init__(self):
#         super().__init__('zed_yolov8_node')
#         self.bridge = CvBridge()

#         # Publisher for bounding boxes
#         self.bounding_box_pub = self.create_publisher(Float32MultiArray, '/yolo8/bounding_boxes', 10)
        
#         # YOLOv8 model initialization using ultralytics package
#         self.model = YOLO('yolov8m.pt')  # Load the medium version of YOLOv8

#         # Subscribe to the ZED camera image topic
#         self.subscription = self.create_subscription(
#             Image,
#             '/zed/zed_node/rgb/image_rect_color',  # Update this to your ZED image topic
#             self.image_callback,
#             10
#         )
#         self.subscription  # Prevent unused variable warning

#     def image_callback(self, msg):
#         # Convert ROS Image message to OpenCV image
#         cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
#         # Perform YOLOv8 inference
#         results = self.model(cv_image)
        
#         # Extract detected bounding boxes
#         detections = self.detections_to_custom_box(results)
        
#         # Publish the bounding boxes in Float32MultiArray format
#         for det in detections:
#             bbox_msg = Float32MultiArray()
#             bbox_msg.data = [det['xmin'], det['ymin'], det['xmax'], det['ymax']]
#             self.bounding_box_pub.publish(bbox_msg)

#     def detections_to_custom_box(self, results):
#         output = []
#         for result in results:
#             for bbox in result.boxes:
#                 xmin, ymin, xmax, ymax = bbox.xyxy[0].tolist()  # Convert tensor to list
#                 obj = {
#                     "xmin": float(xmin),  # Top-left X coordinate
#                     "ymin": float(ymin),  # Top-left Y coordinate
#                     "xmax": float(xmax),  # Bottom-right X coordinate
#                     "ymax": float(ymax)   # Bottom-right Y coordinate
#                 }
#                 output.append(obj)
#         return output

# def main(args=None):
#     rclpy.init(args=args)
#     node = ZedYoloV8Node()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()



###########################################################################################################
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
import ros2_numpy as rnp

from std_msgs.msg import String, Float32MultiArray

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
            '/zed/zed_node/left_raw/image_raw_color',  # Adjust this topic name if necessary
            self.callback,
            10)

    def callback(self, data):
        # print("Received image")
        """Callback function to process image and publish annotated images."""
        # Convert the incoming ROS Image message to a NumPy array using ros2_numpy
        array = rnp.numpify(data)
        # print(array.shape)
        # Determine the encoding and adjust the array if necessary
        if data.encoding in ['rgba8', 'bgra8']:  # Check if the image has an alpha channel
            array = array[:, :, :3]  # Remove the alpha channel
        # print(array.shape)
        # Check if there are any subscribers to the detection image topic
        if self.det_image_pub.get_subscription_count() > 0:
            # Run the detection model on the NumPy array (image)
            det_result = self.detection_model(array)
            # print(det_result)

            # Plot the detection results on the image and return it as a NumPy array
            det_annotated = det_result[0].plot(show=False)
            # Convert the annotated NumPy array back to a ROS Image message and publish it
            det_image_msg = rnp.msgify(Image, det_annotated, encoding="rgb8")
            # print(det_image_msg)
            self.det_image_pub.publish(det_image_msg)

            # Prepare and publish bounding box data
            if self.bbox_pub.get_subscription_count() > 0:
                bounding_boxes = Float32MultiArray()
                for box in det_result[0].boxes:
                    # Extract bounding box data (x_min, y_min, x_max, y_max)
                    x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls.cpu().numpy())  # Class ID
                    confidence = float(box.conf.cpu().numpy())  # Confidence score

                    # Add data to the bounding_boxes array
                    bounding_boxes.data.extend([x_min, y_min, x_max, y_max, class_id, confidence])

                self.bbox_pub.publish(bounding_boxes)
        if self.seg_image_pub.get_subscription_count() > 0:
            # Run the segmentation model on the NumPy array (image)
            seg_result = self.segmentation_model(array)
            # Plot the segmentation results on the image and return it as a NumPy array
            seg_annotated = seg_result[0].plot(show=False)
            # Convert the annotated NumPy array back to a ROS Image message and publish it
            seg_image_msg = rnp.msgify(Image, seg_annotated, encoding="rgb8")
            self.seg_image_pub.publish(seg_image_msg)

        if self.classes_pub.get_subscription_count() > 0:
            det_result = self.detection_model(array)
            classes = det_result[0].boxes.cls.cpu().numpy().astype(int)
            names = [det_result[0].names[i] for i in classes]
            self.classes_pub.publish(String(data=str(names)))

def main(args=None):
    rclpy.init(args=args)  # Initialize the rclpy library
    node = UltralyticsNode()  # Create the node

    try:
        rclpy.spin(node)  # Keep the node running and processing callbacks
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()  # Destroy the node on shutdown
        rclpy.shutdown()  # Shutdown the rclpy library

if __name__ == '__main__':
    main()
