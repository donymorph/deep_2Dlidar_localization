#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

class DualOdomToPath(Node):
    def __init__(self):
        super().__init__('dual_odom_to_path')

        # Subscribe to the /odom topic
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Subscribe to the /AI_odom topic
        self.ai_odom_subscriber = self.create_subscription(
            Odometry,
            '/AI_odom',
            self.ai_odom_callback,
            10
        )

        # Publisher for the /path topic (ground truth path)
        self.path_publisher = self.create_publisher(Path, '/path', 10)

        # Publisher for the /ai_path topic (AI path)
        self.ai_path_publisher = self.create_publisher(Path, '/ai_path', 10)

        # Initialize the Path messages
        self.path = Path()
        self.path.header.frame_id = "odom"

        self.ai_path = Path()
        self.ai_path.header.frame_id = "odom"

        self.get_logger().info("Dual Odom to Path node started.")

    def odom_callback(self, msg: Odometry):
        """Callback for the /odom topic."""
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose

        # Append the pose to the path
        self.path.header.stamp = self.get_clock().now().to_msg()
        self.path.poses.append(pose)

        # Publish the path
        self.path_publisher.publish(self.path)
        #self.get_logger().info(f"Published path with {len(self.path.poses)} poses.")

    def ai_odom_callback(self, msg: Odometry):
        """Callback for the /AI_odom topic."""
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose

        # Append the pose to the AI path
        self.ai_path.header.stamp = self.get_clock().now().to_msg()
        self.ai_path.poses.append(pose)

        # Publish the AI path
        self.ai_path_publisher.publish(self.ai_path)
        #self.get_logger().info(f"Published AI path with {len(self.ai_path.poses)} poses.")

def main(args=None):
    rclpy.init(args=args)
    node = DualOdomToPath()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
