#!/usr/bin/env python3
"""
PID Controller Node
Implements PID control for velocity and steering
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64


class PIDController:
    def __init__(self, kp, ki, kd, output_limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits

        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        """Compute PID output"""
        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative

        # Calculate output
        output = p_term + i_term + d_term

        # Apply output limits
        if self.output_limits:
            output = max(min(output, self.output_limits[1]), self.output_limits[0])

        self.prev_error = error
        return output

    def reset(self):
        """Reset PID controller state"""
        self.integral = 0.0
        self.prev_error = 0.0


class PIDControllerNode(Node):
    def __init__(self):
        super().__init__('pid_controller_node')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel_pid', 10)

        # Subscribers
        self.velocity_error_sub = self.create_subscription(
            Float64,
            '/velocity_error',
            self.velocity_error_callback,
            10
        )

        self.steering_error_sub = self.create_subscription(
            Float64,
            '/steering_error',
            self.steering_error_callback,
            10
        )

        # Parameters
        self.declare_parameter('velocity_kp', 1.0)
        self.declare_parameter('velocity_ki', 0.1)
        self.declare_parameter('velocity_kd', 0.05)
        self.declare_parameter('steering_kp', 1.5)
        self.declare_parameter('steering_ki', 0.0)
        self.declare_parameter('steering_kd', 0.1)
        self.declare_parameter('control_frequency', 50.0)

        # Initialize PID controllers
        self.velocity_pid = PIDController(
            self.get_parameter('velocity_kp').value,
            self.get_parameter('velocity_ki').value,
            self.get_parameter('velocity_kd').value,
            output_limits=(-2.0, 2.0)
        )

        self.steering_pid = PIDController(
            self.get_parameter('steering_kp').value,
            self.get_parameter('steering_ki').value,
            self.get_parameter('steering_kd').value,
            output_limits=(-1.57, 1.57)  # ~90 degrees
        )

        self.control_freq = self.get_parameter('control_frequency').value
        self.dt = 1.0 / self.control_freq

        # State variables
        self.velocity_error = 0.0
        self.steering_error = 0.0

        # Timer for control loop
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info('PID Controller Node initialized')

    def velocity_error_callback(self, msg):
        """Receive velocity error"""
        self.velocity_error = msg.data

    def steering_error_callback(self, msg):
        """Receive steering error"""
        self.steering_error = msg.data

    def control_loop(self):
        """Main control loop"""
        # Compute control outputs
        velocity_output = self.velocity_pid.compute(self.velocity_error, self.dt)
        steering_output = self.steering_pid.compute(self.steering_error, self.dt)

        # Publish control command
        cmd_vel = Twist()
        cmd_vel.linear.x = velocity_output
        cmd_vel.angular.z = steering_output

        self.cmd_vel_pub.publish(cmd_vel)


def main(args=None):
    rclpy.init(args=args)
    node = PIDControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
