#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import os
import csv
from datetime import datetime

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector_node')
        
        # Ensure the dataset folder exists
        self.dataset_folder = "dataset"
        os.makedirs(self.dataset_folder, exist_ok=True)
        
        # Customize filenames if desired
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.odom_filename = os.path.join(self.dataset_folder, f"odom_data_{timestamp_str}.csv")
        self.scan_filename = os.path.join(self.dataset_folder, f"scan_data_{timestamp_str}.csv")

        
        # Open CSV files
        self.odom_file = open(self.odom_filename, 'w', newline='')
        self.scan_file = open(self.scan_filename, 'w', newline='')
        
        # Create CSV writers
        self.odom_writer = csv.writer(self.odom_file)
        self.scan_writer = csv.writer(self.scan_file)
        
        # Write headers for odom CSV
        # Storing only twist linear/angular velocity, plus time info for synchronization
        self.odom_writer.writerow([
            'sec',
            'nanosec',
            'pos_x',
            'pos_y',
            'pos_z',
            'orientation_x',
            'orientation_y',
            'orientation.z',
            'linear_x',
            'linear_y',
            'linear_z',
            'angular_x',
            'angular_y',
            'angular_z'
        ])

        # Write headers for scan CSV
        # Skipping intensities, storing everything else (angle_* parameters, ranges, etc.)
        self.scan_writer.writerow([
            'sec',
            'nanosec',
            'ranges'
        ])
        
        # Create subscribers
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        self.get_logger().info('DataCollector node has been started.')

    def odom_callback(self, msg: Odometry):
        """
        Callback for the /odom topic.
        We capture only twist information: linear and angular velocity.
        """
        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        
        pos = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        

        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular

        # Write a row to odom_data.csv
        self.odom_writer.writerow([
            sec,
            nanosec,
            pos.x,
            pos.y,
            pos.z,
            orientation.x,
            orientation.y,
            orientation.z,
            lin.x,
            lin.y,
            lin.z,
            ang.x,
            ang.y,
            ang.z,
        ])

    def scan_callback(self, msg: LaserScan):
        """
        Callback for the /scan topic.
        We record all LaserScan fields except intensities.
        """
        sec = msg.header.stamp.sec
        nanosec = msg.header.stamp.nanosec
        
        # Convert ranges (list of floats) to a space- or comma-separated string
        ranges_str = ' '.join(map(str, msg.ranges))
        
        # Write a row to scan_data.csv
        self.scan_writer.writerow([
            sec,
            nanosec,
            ranges_str
        ])
    
    def destroy_node(self):
        """Clean up resources before shutting down."""
        super().destroy_node()
        self.odom_file.close()
        self.scan_file.close()
        self.get_logger().info('CSV files closed and node destroyed.')

def main(args=None):
    rclpy.init(args=args)
    
    data_collector = DataCollector()
    try:
        rclpy.spin(data_collector)
    except KeyboardInterrupt:
        data_collector.get_logger().info('Keyboard Interrupt (Ctrl-C) detected.')
    finally:
        data_collector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
