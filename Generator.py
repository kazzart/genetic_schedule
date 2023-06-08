import numpy as np
import random

from enums import RoomType, Weekday, ClassType


def generate_schedule(z, a, teachers, groups):
    t = []  # Periods
    for week_day in range(1, 7):
        for time_period in range(7):
            t.append({"weekday": Weekday(week_day),
                     "time_period": time_period})

    teacher_ij = np.zeros((len(teachers), 6, 7))
    group_ij = np.zeros((len(groups), 6, 7))
    room_ij = np.zeros((len(a), 6, 7))

    alpha = np.array([])
    tau = np.array([])

    for class_i in z:
        a_bts = np.array([])
        if class_i["room_type"] in (RoomType.B, RoomType.T):
            for idx, room in enumerate(a):
                if room["room_type"] == class_i["room_type"]:
                    a_bts = np.append(a_bts, idx)
            alpha_ch = int(random.choice(a_bts))
        else:
            for idx, room in enumerate(a):
                if room["room_type"] in (RoomType.B, RoomType.S):
                    a_bts = np.append(a_bts, idx)
            alpha_ch = int(random.choice(a_bts))
        tau_ch = random.choice(range(len(t)))

        counter = 1
        while teacher_ij[class_i['teacher']][t[tau_ch]['weekday'].value - 1][t[tau_ch]['time_period']] or group_ij[class_i['group']][t[tau_ch]['weekday'].value - 1][t[tau_ch]['time_period']] or room_ij[alpha_ch][t[tau_ch]['weekday'].value - 1][t[tau_ch]['time_period']]:
            a_bts = np.array([])
            if class_i["room_type"] in (RoomType.B, RoomType.T):
                for idx, room in enumerate(a):
                    if room["room_type"] == class_i["room_type"]:
                        a_bts = np.append(a_bts, idx)
                alpha_ch = int(random.choice(a_bts))
            else:
                for idx, room in enumerate(a):
                    if room["room_type"] in (RoomType.B, RoomType.S):
                        a_bts = np.append(a_bts, idx)
                alpha_ch = int(random.choice(a_bts))
            tau_ch = random.choice(range(len(t)))

            # print(class_i['teacher'], t[tau_ch]['weekday'].value - 1,
            #       t[tau_ch]['time_period'], class_i['group'], alpha_ch)
            # print(teacher_ij[class_i['teacher']][t[tau_ch]['weekday'].value - 1][t[tau_ch]['time_period']], group_ij[class_i['group']]
            #       [t[tau_ch]['weekday'].value - 1][t[tau_ch]['time_period']], room_ij[alpha_ch][t[tau_ch]['weekday'].value - 1][t[tau_ch]['time_period']])
            # print(teacher_ij[class_i['teacher']][t[tau_ch]['weekday'].value - 1][t[tau_ch]['time_period']] or group_ij[class_i['group']]
            #       [t[tau_ch]['weekday'].value - 1][t[tau_ch]['time_period']] or room_ij[alpha_ch][t[tau_ch]['weekday'].value - 1][t[tau_ch]['time_period']])
            counter += 1
            # if counter % 5 == 0:
            #     print(f'Generating iteration {counter}')
            if counter == 100:
                raise Exception(f'Failed to generate for {class_i}')
            if not (teacher_ij[class_i['teacher']][t[tau_ch]['weekday'].value - 1][t[tau_ch]['time_period']] or group_ij[class_i['group']][t[tau_ch]['weekday'].value - 1][t[tau_ch]['time_period']] or room_ij[alpha_ch][t[tau_ch]['weekday'].value - 1][t[tau_ch]['time_period']]):
                break

        teacher_ij[class_i['teacher']][t[tau_ch]['weekday'].value - 1
                                       ][t[tau_ch]['time_period']] = 1
        group_ij[class_i['group']][t[tau_ch]['weekday'].value - 1
                                   ][t[tau_ch]['time_period']] = 1
        room_ij[alpha_ch][t[tau_ch]
                          ['weekday'].value - 1][t[tau_ch]['time_period']] = 1
        alpha = np.append(alpha, a[alpha_ch])
        tau = np.append(tau, t[tau_ch])
    return alpha, tau
