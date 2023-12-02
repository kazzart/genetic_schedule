import pandas as pd
import numpy as np
from enums import Weekday
from Random_constants import TIME_PERIODS, TEACHERS, GROUPS


def output_df(classes: list, alpha: np.ndarray, tau: np.ndarray):
    """
    Generate a schedule dataframe based on the given classes, alpha values, and tau values.

    Args:
        - classes (list): A list of dictionaries representing different classes. Each dictionary contains the keys 'group', 'discipline', 'class_type', and 'teacher'.
        - alpha (np.ndarray): A numpy array of alpha values.
        - tau (np.ndarray): A numpy array of tau values. Each tau value is a dictionary with the keys 'weekday' and 'time_period'.

    Returns:
        - pd.DataFrame: A pandas dataframe representing the schedule for different groups and time periods.
    """
    schedule = {}
    t = []  # Periods
    for week_day in range(1, 7):
        for time_period in range(7):
            t.append((Weekday(week_day).name, TIME_PERIODS[time_period]))
    index = pd.MultiIndex.from_tuples(t, names=["Day", "Time"])
    for idx, class_i in enumerate(classes):

        group = GROUPS[class_i['group']]
        g_schedule = schedule.get(group, ['-'] * (6*7))
        room = alpha[idx]
        class_i['room'] = room
        period = tau[idx]
        day = period['weekday'].value - 1
        time = period['time_period']
        g_schedule[day * 7 +
                   time] = f"{class_i['discipline']} {room['room_number']} {class_i['class_type'].name} {TEACHERS[class_i['teacher']]}"
        schedule[group] = g_schedule
    print(schedule)

    return pd.DataFrame(schedule, index=index)
