#!/usr/bin/env python

import time
import math
import rospy
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from tf.transformations import euler_from_quaternion

class Bobocar:
    def __init__(self):
        self.move_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

    def control_movement(self, linear_vel, angular_vel):
        twist = Twist()
        twist.linear.x = max(min(linear_vel, 0.03), 0.06)
        twist.angular.z = max(min(angular_vel, 0.2), -0.1)
        self.move_pub.publish(twist)
        return twist.linear.x, twist.angular.z

    def stop(self):
        twist = Twist()
        twist.linear.x = 0
        twist.angular.z = 0
        self.move_pub.publish(twist)
        return twist.linear.x, twist.angular.z

class BobocarController:
    def __init__(self, Kp_linear=0.8, Ki_linear=0.001, Kd_linear=0.001, 
                 Kp_angular=0.8, Ki_angular=0.001, Kd_angular=0.001,
                 reach_threshold=0.1):
        self.car = Bobocar()
        
        # 다음 목표 좌표 요청을 위한 퍼블리셔 초기화
        self.request_next_target_pub = rospy.Publisher('request_next_target', Bool, queue_size=10)
        
        # PID parameters
        self.Kp_linear = Kp_linear
        self.Ki_linear = Ki_linear
        self.Kd_linear = Kd_linear
        self.Kp_angular = Kp_angular
        self.Ki_angular = Ki_angular
        self.Kd_angular = Kd_angular
        
        self.reach_threshold = reach_threshold

        # Initial state
        self.initialized = False
        self.initial_yaw = 0.0
        self.init_position()  # 초기 위치 설정 함수 호출
        self.odom_position = 0.0
        self.current_angle = 0.0
        self.target_position = None  # 목표 좌표

        # Target variables
        self.linear_error_sum, self.linear_last_error = 0.0, 0.0
        self.angular_error_sum, self.angular_last_error = 0.0, 0.0
        self.last_time = None

    def init_position(self):
        # 고정된 초기 좌표를 설정
        self.current_position = (640, 720)

    def set_target(self, x, y):  # center_point_callback에서 호출
        self.target_position = (x, y)

    def update_odom_position(self, position):
        # odom 데이터를 통해 업데이트되는 실제 위치
        self.odom_position = position

    def update_current_position(self):
        # 목표 도달 시 현재 위치를 업데이트
        if self.target_position:
            self.current_position = self.target_position
            self.target_position = None

    def update_current_angle(self, angle):
        if not self.initialized:
            self.initial_yaw = angle
            self.current_angle = 0.0
        else:
            self.current_angle = angle - self.initial_yaw

    def initialize(self):
        self.initialized = True

    def control(self):
        if self.target_position is None:
            return  # 목표가 설정되지 않은 경우 제어 수행하지 않음

        # 초기 현재 좌표와 목표 좌표의 거리 및 각도 계산
        if not self.initialized:
            start_x, start_y = self.current_position
        else:
            start_x, start_y = self.init_position  # 초기화 후에는 odom 위치를 사용

        target_x, target_y = self.target_position
        distance_to_target = math.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)
        target_theta = math.atan2(target_y - start_y, target_x - start_x)
        angle_diff = target_theta - self.current_angle  # 목표 각도와 현재 각도 차이

        # 거리 및 각도 오차 계산
        linear_error = distance_to_target - self.odom_position[0]  # odom의 x 값을 사용하여 거리 오차 계산
        angular_error = angle_diff - self.current_angle  # 남은 각도 오차

        # 목표 도달 여부 확인
        if distance_to_target < self.reach_threshold:
            rospy.loginfo("목표에 도달했습니다. 정지합니다.")
            self.car.stop()
            # 현재 위치를 최신 목표로 업데이트
            self.update_current_position()  # 목표를 현재 위치로 설정
            # 다음 목표 요청
            self.request_next_target_pub.publish(Bool(data=True))  # 다음 목표 요청 발행
            return

        # 시간 간격 계산
        current_time = time.time()
        if self.last_time is None:
            self.last_time = current_time
            return
        dt = current_time - self.last_time
        self.last_time = current_time

        # 직선 PID 제어
        self.linear_error_sum += linear_error * dt
        linear_derivative = (linear_error - self.linear_last_error) / dt
        self.linear_last_error = linear_error
        linear_velocity = (self.Kp_linear * linear_error +
                           self.Ki_linear * self.linear_error_sum +
                           self.Kd_linear * linear_derivative)
        linear_velocity = max(min(abs(linear_velocity), 0.03), 0.06)

        # 조향 PID 제어
        self.angular_error_sum += angular_error * dt
        angular_derivative = (angular_error - self.angular_last_error) / dt
        self.angular_last_error = angular_error
        angular_velocity = (self.Kp_angular * angular_error +
                            self.Ki_angular * self.angular_error_sum +
                            self.Kd_angular * angular_derivative)
        angular_velocity = max(min(angular_velocity, 0.2), -0.1)

        # 이동 제어 명령 발행
        self.car.control_movement(linear_velocity, angular_velocity)
        rospy.loginfo(f"Distance to target: {distance_to_target}, angle difference: {angle_diff}")

def odom_callback(data, controller):
    position = (data.pose.pose.position.x, data.pose.pose.position.y)
    controller.update_odom_position(position)  # odom의 실제 위치 업데이트

    orientation_q = data.pose.pose.orientation
    _, _, yaw = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
    controller.update_current_angle(yaw)

    if not controller.initialized:
        controller.initialize()

def center_point_callback(data, controller):
    rospy.loginfo(f"Received target point: ({data.x}, {data.y}, {data.z})")
    controller.set_target(data.x, data.y)

def main():
    rospy.init_node('control_node')
    
    controller = BobocarController()
    rospy.Subscriber('odom', Odometry, odom_callback, controller)
    rospy.Subscriber('center_point', Point, center_point_callback, controller)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        controller.control()
        rate.sleep()

if __name__ == '__main__':
    main()
