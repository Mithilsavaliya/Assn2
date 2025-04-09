import gymnasium as gym
from gymnasium import spaces
import numpy as np

# ROS 2 & OpenCV
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2

class RedBall(Node):
    def __init__(self):
        super().__init__('redball_node')

        self.subscription = self.create_subscription(
            Image,
            'custom_ns/camera1/image_raw',
            self.image_callback,
            10)
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        self.br = CvBridge()
        self.latest_x = 320  # Default center
        self.image_received = False

    def image_callback(self, msg):
        try:
            frame = self.br.imgmsg_to_cv2(msg)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower_red = (110, 100, 100)
            upper_red = (130, 255, 255)
            mask = cv2.inRange(hsv, lower_red, upper_red)

            blurred = cv2.GaussianBlur(mask, (9, 9), 2)
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 150,
                                       param1=100, param2=20, minRadius=2, maxRadius=2000)

            if circles is not None:
                circle = circles[0][0]
                self.latest_x = int(circle[0])
            else:
                self.latest_x = 320  # fallback to center
            self.image_received = True
        except Exception as e:
            self.get_logger().warn(f"Error in image_callback: {e}")

    def get_redball_x(self):
        return self.latest_x

class CreateRedBallEnv(gym.Env):
    def __init__(self):
        super().__init__()

        rclpy.init()
        self.redball_node = RedBall()

        self.observation_space = spaces.Discrete(640)  # x-axis position
        self.action_space = spaces.Discrete(3)  # [0: left, 1: no turn, 2: right]

        self.step_count = 0
        self.max_steps = 100

    def step(self, action):
        # Publish a Twist command based on action
        twist = Twist()
        if action == 0:
            twist.angular.z = 0.5  # Turn left
        elif action == 1:
            twist.angular.z = 0.0  # No turn
        elif action == 2:
            twist.angular.z = -0.5  # Turn right
        self.redball_node.publisher.publish(twist)

        # Spin ROS node once to update image callback
        rclpy.spin_once(self.redball_node)

        obs = self.redball_node.get_redball_x()
        reward = 0
        done = False

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        rclpy.spin_once(self.redball_node)
        obs = self.redball_node.get_redball_x()
        return obs, {}

    def render(self):
        return  # No human rendering

    def close(self):
        self.redball_node.destroy_node()
        rclpy.shutdown()
