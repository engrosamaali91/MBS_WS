#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image , CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import PoseStamped, Pose
# from tf2_geometry_msgs import PoseStamped
import numpy as np
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs 

from std_msgs.msg import Float32MultiArray

class ZED2iSubscriber(Node):
    def __init__(self):
        super().__init__('zed2i_subsriber')

        #define the subscriber with zed2i camera topics

        self.camera_info_sub = Subscriber(self, CameraInfo, '/zed/zed_node/depth/camera_info')
        self.depth_image_sub = Subscriber(self, Image, '/zed/zed_node/depth/depth_registered')
        self.bounding_box_sub = self.create_subscription(Float32MultiArray, '/ultralytics/detection/bounding_boxes', self.bounding_box_callback, 10)
        self.latest_bounding_box = None

        #synchornize the camera info and depth image
        self.sync = ApproximateTimeSynchronizer([self.camera_info_sub, self.depth_image_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.synced_callback)

        self.bridge = CvBridge()
        # TF buffer and listener for transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)


    def bounding_box_callback(self, msg):   
        if msg.data:
            self.latest_bounding_box = (int(msg.data[0]), int(msg.data[1]), int(msg.data[2]), int(msg.data[3]))
        else:
            self.get_logger().warn("Received bounding box with invalid data length")

    def synced_callback(self, camera_info_msg, depth_msg):
        if self.latest_bounding_box is None:
            self.get_logger().warn("No bounding box received yet")
            return
        # self.get_logger().info(f'Received camera info: {camera_info_msg} and depth image: {depth_image_msg}')
        camera_matrix = self.get_camera_matrix(camera_info_msg)
        # print(camera_matrix)
        camera_frame = camera_info_msg.header.frame_id
        # print(camera_frame)
        world_frame = 'map' #define the target frame for the 3d point 

        #convert the depth image to OpenCV format
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1").astype(np.float32)
        # print(depth_image)
        # depth_image = depth_image / 1000.0 # convert depth values from mm to meters
        #the shape of depth image is 720x1280
        #define the point you want to convert (e.g. the center of the image)
        # height, width = depth_image.shape
        bbox_top_left = (self.latest_bounding_box[0], self.latest_bounding_box[1])
        bbox_bottom_right = (self.latest_bounding_box[2], self.latest_bounding_box[3])
        midpoint = ((bbox_top_left[0] + bbox_bottom_right[0])//2, (bbox_top_left[1] + bbox_bottom_right[1])//2)
        # midpoint = (width//2, height//2)
        # print(midpoint)

        # depth_array = np.array(depth_image, dtype=np.float32)

        #get the depth value at the midpoint
        depth_value = depth_image[midpoint[1], midpoint[0]]
        print('Distance from the object in meters: {}'.format(depth_value))


        # convert pixel to 3d point in camera frame
        point_3d = self.pixel_to_3d(midpoint, depth_value, camera_matrix)

        # print('3D point in camera frame: {}'.format(point_3d))

        #convert the 3d point from the camera fram to the world frame

        world_pose = self.camera_to_world(point_3d, camera_frame, world_frame)
        print(world_pose)

        
    def camera_to_world(self, camera_point, camera_frame, world_frame):
        #compute the transform from the camera frame to the world frame
        # pose = Pose()
        # pose.position.x = float(camera_point[0])
        # pose.position.y = float(camera_point[1])
        # pose.position.z = float(camera_point[2])
        # pose.orientation.w = 1.0

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = camera_frame 
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        # pose_stamped.pose = pose

        pose_stamped.pose.position.x = float(camera_point[0])
        pose_stamped.pose.position.y = float(camera_point[1])
        pose_stamped.pose.position.z = float(camera_point[2])
        pose_stamped.pose.orientation.w = 1.0  # Assuming no rotation


        # # print(pose_stamped)

        try:
            #get the transform from the camera frame to the world frame
            transform = self.tf_buffer.lookup_transform(world_frame, camera_frame, rclpy.time.Time(), rclpy.time.Duration(seconds=1.0))
        # print(transform)
            transformed_pose = tf2_geometry_msgs.do_transform_pose(pose_stamped.pose, transform)
        # print(transformed_pose)
            return transformed_pose
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.get_logger().warn("Could not get transform from {} to {}".format(camera_frame, world_frame))
            return None

    def pixel_to_3d(self, point, depth, K):
        """
        Converts a pixel coordinate to a 3D point in the camera frame.

        Args:
            point (tuple): The pixel coordinate (u, v).
            depth (float): The depth value corresponding to the pixel.
            K (list): The camera intrinsic matrix.

        Returns:
            tuple: The 3D point in the camera frame (x, y, z).

        Raises:
            ValueError: If the depth value is not positive.
        """
        K = np.array(K).reshape(3, 3)

        u, v = point
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # Handle non-positive depth values
        if depth <= -1:
            raise ValueError("Invalid depth value, depth must be positive")

        # Compute the 3D point in the camera frame
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return (x, y, z)
    def get_camera_matrix(self, camera_info):
         #convert k i.e 1d array to 3x3 matrix
        return np.array(camera_info.k).reshape(3, 3)

def main(args=None):
    rclpy.init(args=args)

    node = ZED2iSubscriber()

    try: 
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()