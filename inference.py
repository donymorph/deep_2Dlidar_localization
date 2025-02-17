#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header

import torch
import numpy as np
from architectures.architectures import (
    SimpleMLP,
    Conv1DNet,
    Conv1DLSTMNet,
    ConvTransformerNet,
    CNNTransformerNet_Optuna,
    CNNLSTMNet_Optuna

)
import os
# For TF broadcasting
from tf2_ros import TransformBroadcaster
import tf_transformations  # e.g., quaternion_from_euler


###################################
# Main Inference Node
###################################
class ScanToOdomInferenceNode(Node):
    def __init__(self):
        super().__init__('scan_to_odom_inference_node')

        ############################
        # Parameters / State
        ############################
        self.model_initialized = False
        self.model = None
        self.model_name = 'CNNLSTMNet_Optuna_lr0.0006829381720401536_bs16_20250127_144344.pth' # change it according to name of model
        # Robot 2D state: x, y, yaw
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_stamp = None
        # ---------------------------
        # Ensure dataset folder exists and retrieve files
        # ---------------------------
        models_folder = "models"
        if not os.path.exists(models_folder):
            raise FileNotFoundError(f"The models folder '{models_folder}' does not exist. Please create it and add the model files.")

        # Set paths to dataset files inside the dataset folder
        self.model_path = os.path.join(models_folder, self.model_name)


        ############################
        # TF broadcaster
        ############################
        self.tf_broadcaster = TransformBroadcaster(self)

        ############################
        # Publishers / Subscribers
        ############################
        self.odom_pub = self.create_publisher(Odometry, '/AI_odom', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.get_logger().info("ScanToOdomInferenceNode initialized. Awaiting LiDAR data...")


    def initialize_model(self, input_size: int):
        """
        Initialize and load your trained model once we know how many LiDAR beams exist.
        """
        #self.model = SimpleMLP(input_size=input_size, output_size=6)
        self.model = CNNLSTMNet_Optuna(input_size = input_size, output_size=3)
        # Load weights
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu'))) # cuda
        self.model.eval()
        self.model_initialized = True

        self.get_logger().info(f"Model loaded from '{self.model_path}' with input_size={input_size}.")

    def scan_callback(self, scan_msg: LaserScan):
        """
        Callback that receives a LaserScan message, runs inference, and publishes Odometry.
        """
        # 1. Convert LaserScan ranges into a NumPy or PyTorch tensor
        ranges = list(scan_msg.ranges)  # type: ignore
        # If you used the entire ranges array during training, you must keep it consistent
        input_data = np.array(ranges, dtype=np.float32)

        # If model is not yet initialized, do it now based on current scan length
        if not self.model_initialized:
            input_size = len(input_data)
            self.initialize_model(input_size)

        # 2. Forward pass through the model
        #    shape: [1, input_size] for a single sample
        input_tensor = torch.from_numpy(input_data).unsqueeze(0)  # shape (1, N)
        with torch.no_grad():
            pred = self.model(input_tensor)  # shape (1, 6)

        # 3. Extract predicted velocities
        #    pred[0] is [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
        #linear_x, linear_y, angular_z = pred[0].tolist()
        pos_x, pos_y, orientation_z = pred[0].tolist()
        # 2) Compute time delta (dt) to integrate velocities
        current_time = scan_msg.header.stamp
        current_seconds = current_time.sec + current_time.nanosec * 1e-9

        if self.last_stamp is None:
            # First callback - no integration yet
            self.last_stamp = current_time
            return

        last_seconds = self.last_stamp.sec + self.last_stamp.nanosec * 1e-9
        dt = current_seconds - last_seconds
        self.last_stamp = current_time

        # 3) Numerical integration
        # x, y updated by predicted linear velocities in the odom frame
        # self.x += linear_x * dt
        # self.y += linear_y * dt
        # self.yaw += angular_z * dt  # only about z in 2D plane

        # Convert yaw to quaternion
        q = tf_transformations.quaternion_from_euler(0.0, 0.0, self.yaw)

        # 4a) Publish TF transform (odom -> base_link)
        t = TransformStamped()
        t.header.stamp = current_time
        t.header.frame_id = "odom"
        t.child_frame_id = "AI_base_link"
        t.transform.translation.x = pos_x
        t.transform.translation.y = pos_y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = orientation_z #q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

        # 4b) Publish Odometry message
        odom_msg = Odometry()
        odom_msg.header = Header()
        odom_msg.header.stamp = current_time
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "AI_base_link"

        # Pose (x, y, yaw)
        odom_msg.pose.pose.position.x = pos_x
        odom_msg.pose.pose.position.y = pos_y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = orientation_z #q[2]
        odom_msg.pose.pose.orientation.w = q[3]

        # Twist (linear_x, linear_y, angular_z)
        odom_msg.twist.twist.linear.x = 0.0 #pos_x
        odom_msg.twist.twist.linear.y = 0.0 #pos_y
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = 0.0

        self.odom_pub.publish(odom_msg)

        # Optional debug logging
        # self.get_logger().debug(
        #     f"dt={dt:.3f}, pred_vel=({linear_x:.3f},{linear_y:.3f},{angular_z:.3f}), "
        #     f"pose=({self.x:.3f},{self.y:.3f},{self.yaw:.3f})"
        # )

def main(args=None):
    rclpy.init(args=args)
    node = ScanToOdomInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
