"""Main module"""

import numpy as np

from Schedule import Schedule
from GeneticAlgorithm import GeneticAlgorithm

from Output import output_df
import matplotlib.pyplot as plt

from enums import RoomType, Weekday, ClassType
from Random_constants import (
    DISCIPLINES,
    GROUPS,
    TEACHERS,
    ROOM_NUMBERS,
    TIME_PERIODS,
    DISCIPLINES_TEACHERS,
)

# DISCIPLINES = np.array(['ССПРП'])
# GROUPS = np.array(['ИКМО-03-22'])
# TEACHERS = np.array(['Демидова Лилия Анатолиевна'])
# ROOM_NUMBERS = np.array(['Г-101'])
# TIME_PERIODS = np.array([
#      '9:00 - 10:30',
#      '10:40 - 12:10',
#      '12:40 - 14:10',
#      '14:40 - 16:10',
#      '16:20 - 17:50',
#      '18:00 - 19:30',
#      '19:40 - 21:10'
#     ])


z = []  # Classes
a = []  # Rooms
# t = [] # Periods

# Solution vectors
alpha = np.array([])  # Room for i-th class
tau = np.array([])  # Period for i-th class

# Example
class1 = {
    "teacher": TEACHERS[0],
    "discipline": DISCIPLINES[0],
    "group": GROUPS[0],
    "room_type": RoomType.S,
    "class_type": ClassType.LECTURE,
}
room1 = {"room_number": ROOM_NUMBERS[0], "room_type": RoomType.S}
# period1 = np.array([Weekday.TUESDAY, TIME_PERIODS[7]])

# for week_day in range(1, 7):
#     for time_period in range(7):
#             t.append({'weekday': Weekday(week_day), 'time_period': time_period})

for group in range(len(GROUPS)):
    group_disciplines = np.random.choice(DISCIPLINES, 6, replace=False)
    for discipline in group_disciplines:
        discipline_teachers = np.random.choice(
            DISCIPLINES_TEACHERS[discipline], 3, replace=True
        )
        class1 = {
            "teacher": discipline_teachers[0],
            "discipline": discipline,
            "group": group,
            "room_type": RoomType.B,
            "class_type": ClassType.LECTURE,
        }
        class2 = {
            "teacher": discipline_teachers[1],
            "discipline": discipline,
            "group": group,
            "room_type": RoomType.T,
            "class_type": ClassType.LABORATORY,
        }
        class3 = {
            "teacher": discipline_teachers[2],
            "discipline": discipline,
            "group": group,
            "room_type": RoomType.S,
            "class_type": ClassType.PRACTICAL,
        }
        z.append(class1)
        z.append(class2)
        z.append(class3)

for idx, room in enumerate(ROOM_NUMBERS):
    a.append({"room_number": room, "room_type": RoomType.B if idx < 4 else RoomType.T if idx < 8 else RoomType.S})

# z.append(class1)
# z.append(class1)
# z.append(class1)
# z.append(class1)
# z.append(class1)
# z.append(class1)
# z.append(class1)
# z.append(class1)
# a.append(room1)
# t = np.append(t, period1)

print(z, a)

init_args = {
    "classes": z,
    "rooms": a,
    "teachers": TEACHERS,
    "groups": GROUPS,
    "number_of_classes": len(z),
    "alpha": None,
    "tau": None,
    "number_of_genes_to_mutate": 1,
}
ga = GeneticAlgorithm(Schedule, 200, 0.1, init_args, {}, max_iter=3000, max_iter_no_improve=150)  # type: ignore

ga.run()
# print(ga.best_solution.conflict) # type: ignore
# print(ga.best_solution.alpha) # type: ignore
# print(ga.best_solution.tau) # type: ignore
# list_week = [[], [], [], [], [], []]
# for idx, period in enumerate(ga.best_solution.tau):
#     list_week[period['weekday']].append(z[idx])
# print(list_week)
df = output_df(z, ga.best_solution.alpha, ga.best_solution.tau) # type: ignore
history = list(zip(*ga.history))
print(history)
plt.plot(history[0], label='Conflicts') # type: ignore
plt.plot(history[1], label='Quality') # type: ignore
plt.legend(loc="lower right")
plt.xlabel('Iteration')
plt.ylabel('Points')
plt.title('Fitness history')
# plt.show()
plt.savefig('figure.png')
# print(ga.history)
df.to_excel('Schedule.xlsx')
