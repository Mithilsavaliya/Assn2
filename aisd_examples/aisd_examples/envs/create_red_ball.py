import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from irobot_create_msgs.msg import StopStatus
from cv_bridge import CvBridge
import cv2

class RedBall(Node):
    def __init__(self):
        super().__init__('redball_node')
        self.br = CvBridge()
        self.latest_x = 320  # Default middle if no ball detected
        self.create3_is_stopped = True

        self.image_sub = self.create_subscription(
            Image,
            '/custom_ns/camera1/image_raw',
            self.image_callback,
            10)

        self.stop_sub = self.create_subscription(
            StopStatus,
            '/stop_status',
            self.stop_callback,
            10)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def image_callback(self, msg):
        frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = (110, 100, 100)
        upper_red = (130, 255, 255)
        mask = cv2.inRange(hsv, lower_red, upper_red)

        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 150, param1=100, param2=20, minRadius=2, maxRadius=2000)
        if circles is not None:
            x = int(circles[0][0][0])
            self.latest_x = x
        else:
            self.latest_x = 320  # Default to center if ball not detected

    def stop_callback(self, msg):
        self.create3_is_stopped = msg.is_stopped

    def step(self, action):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = (action - 320) / 320 * (np.pi / 2)
        self.create3_is_stopped = False
        self.cmd_vel_pub.publish(twist)

class CreateRedBallEnv(gym.Env):
    def __init__(self):
        super().__init__()
        rclpy.init()
        self.redball = RedBall()
        self.observation_space = spaces.Discrete(640)
        self.action_space = spaces.Discrete(640)
        self.step_count = 0

    def reset(self, seed=None, options=None):
        self.step_count = 0
        return self.redball.latest_x, {}

    def step(self, action):
        self.redball.step(action)
        rclpy.spin_once(self.redball)
        while not self.redball.create3_is_stopped:
            rclpy.spin_once(self.redball)
        self.step_count += 1

        obs = self.redball.latest_x
        reward = self.compute_reward(obs)
        done = self.step_count >= 100
        return obs, reward, done, False, {}

    def compute_reward(self, x):
        # reward is max at center, min at edges
        return -abs(x - 320) / 320

    def render(self):
        pass  # no human rendering

    def close(self):
        self.redball.destroy_node()
        rclpy.shutdown()
