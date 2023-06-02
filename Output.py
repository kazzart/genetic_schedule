import pandas as pd
import numpy as np
from enums import Weekday
from Random_constants import TIME_PERIODS, TEACHERS




def output_df(classes: list, alpha: np.ndarray, tau: np.ndarray):
    schedule = {}
    t = []  # Periods
    for week_day in range(1, 7):
        for time_period in range(7):
            t.append((Weekday(week_day).name, TIME_PERIODS[time_period]))
    index = pd.MultiIndex.from_tuples(t, names=["Day", "Time"])
    for idx, class_i in enumerate(classes):
        
        group = class_i['group']
        g_schedule = schedule.get(group, ['-'] * (6*7))
        room = alpha[idx]
        class_i['room'] = room
        period = tau[idx]
        day = period['weekday'].value - 1
        time = period['time_period']
        g_schedule[day * 7 + time] = f"{class_i['discipline']} {room['room_number']} {class_i['class_type'].name} {TEACHERS[class_i['teacher']]}"
        schedule[group] = g_schedule
    print(schedule)
    
    return pd.DataFrame(schedule, index=index)
