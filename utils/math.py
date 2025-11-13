import numpy as np


def rotate_2d(points, yaw_degrees):
    yaw_radians = np.deg2rad(yaw_degrees)
    return rotate_2d_radians(points, yaw_radians)


def rotate_2d_radians(points, yaw_radians):
    c, s = np.cos(yaw_radians), np.sin(yaw_radians)
    rotate_R = np.array(((c, -s), (s, c)))
    return points @ rotate_R


def clamp_angle_deg(a):
    return (a + 180) % 360 - 180


def factorize(x, start=2):
    final = int(np.floor(np.sqrt(x)))
    for divisor in range(start, final + 1):
        quotient = x // divisor
        x_no_remainder = quotient * divisor
        if x_no_remainder == x:
            return factorize(quotient, start=divisor) + [
                divisor,
            ]
    return [
        x,
    ]
