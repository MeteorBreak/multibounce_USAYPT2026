import math

def calculateVelocity(deltatime, _diameter = 0.04):
    deltatime_in_second = deltatime / 1000
    v = _diameter / deltatime_in_second
    return v

def degreesToRadians(angle):
    angle_in_radians = angle * math.pi / 180
    return angle_in_radians

g = 9.80665

while(input("start a new calculation? (y/n)") == "y"):

    theta, distance, time1, time2 = input("input in order: theta(in degree) distance time1 time2\n").split()

    theta = float(theta)
    distance = float(distance)
    time1 = float(time1)
    time2 = float(time2)

    # The ball accelerates down the slope, so the time to pass the second gate
    # should be shorter. We swap them to make sure time1 is the shorter time
    # (final velocity) and time2 is the longer time (initial velocity).
    if time1 > time2:
        time1, time2 = time2, time1

    v_final = calculateVelocity(time1)
    v_initial = calculateVelocity(time2)

    mu = math.tan(degreesToRadians(theta)) - (v_final**2 - v_initial**2) / (2*distance*g*math.cos(degreesToRadians(theta)))

    print(mu)
