import math
import numpy as np


# returns angle between three points
def calculateAngle(a, b, c):
    x1 = a.x
    y1 = a.y
    x2 = b.x
    y2 = b.y
    x3 = c.x
    y3 = c.y
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360

    return angle


"""
calculates the coordinates of a new point based on a rotation and 
scaling transformation applied to an existing line segment
"""
def computePoint(x1, y1, x2, y2, theta, k):
    dx = x1 - x2
    dy = y1 - y2
    theta = math.radians(theta)
    rdx = dx * math.cos(theta) - dy * math.sin(theta)
    rdy = dx * math.sin(theta) + dy * math.cos(theta)
    ab = math.sqrt(dx*dx + dy*dy)
    return (k/ab * rdx + x2, k/ab *rdy + y2)

def calculateDistance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
