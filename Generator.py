import numpy as np
import random

from enums import RoomType, Weekday, ClassType


def generate_schedule(z, a):
    t = []  # Periods
    for week_day in range(1, 7):
        for time_period in range(7):
            t.append({"weekday": Weekday(week_day), "time_period": time_period})

    alpha = np.array([])
    tau = np.array([])

    for class_i in z:
        a_bts = np.array([])
        if class_i["room_type"] in (RoomType.B, RoomType.T):
            for room in a:
                if room["room_type"] == class_i["room_type"]:
                    a_bts = np.append(a_bts, room)
            alpha = np.append(alpha, random.choice(a_bts))
        else:
            for room in a:
                if room["room_type"] in (RoomType.B, RoomType.S):
                    a_bts = np.append(a_bts, room)
            alpha = np.append(alpha, random.choice(a_bts))
        tau = np.append(tau, random.choice(t))

    return alpha, tau
