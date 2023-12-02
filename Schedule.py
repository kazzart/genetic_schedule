import numpy as np
from GeneticAlgorithm import Chromosome
from enums import RoomType, Weekday, ClassType
from Generator import generate_schedule


class Schedule(Chromosome):
    """
The `Schedule` class represents a schedule for a set of classes. It is a subclass of the `Chromosome` class and contains methods for calculating the fitness of the schedule, breeding new schedules, and mutating existing schedules.

Example Usage:
    # Create a schedule
    schedule = Schedule(classes, rooms, teachers, groups, number_of_classes)

    # Calculate the fitness of the schedule
    fitness = schedule.fitness()

    # Breed two schedules
    offspring = schedule.breed(other_schedule)

    # Mutate a schedule
    schedule.mutate()

    # Create a copy of a schedule
    copy_schedule = schedule.copy()

Methods:
    __init__(self, classes, rooms, teachers, groups, number_of_classes, alpha=None, tau=None, number_of_genes_to_mutate=1): 
        Initializes a new instance of the `Schedule` class with the given parameters. If alpha and tau are not provided, they are generated using the `generate_schedule` function.
    
    conflicts(self, class_num_i, class_num_j): 
        Calculates the number of conflicts between two classes based on their alpha and tau values.
    
    fitness(self): 
        Calculates the fitness of the schedule based on conflicts and quality criteria.
    
    breed(self, other): 
        Breeds two schedules by combining their alpha and tau vectors.
    
    mutate(self): 
        Mutates the schedule by randomly changing the alpha and tau vectors.
    
    copy(self): 
        Creates a copy of the schedule.

Fields:
    classes: A list of class objects.
    rooms: A list of room objects.
    teachers: A list of teacher objects.
    groups: A list of group objects.
    number_of_classes: The number of classes in the schedule.
    number_of_genes_to_mutate: The number of genes to mutate when performing a mutation.
    alpha: The room for each class.
    tau: The period for each class.
"""
    def __init__(
        self,
        classes: list,
        rooms: list,
        teachers: list,
        groups: list,
        number_of_classes: int,
        alpha: np.ndarray | None = None,
        tau: np.ndarray | None = None,
        number_of_genes_to_mutate: int = 1,
    ) -> None:
        # Solution vectors
        self.classes = classes
        self.rooms = rooms
        self.teachers = teachers
        self.groups = groups
        self.number_of_classes = number_of_classes
        self.number_of_genes_to_mutate = number_of_genes_to_mutate
        self.t = []  # Periods
        for week_day in range(1, 7):
            for time_period in range(7):
                self.t.append({"weekday": Weekday(week_day),
                               "time_period": time_period})
        if alpha is None or tau is None:
            self.alpha, self.tau = generate_schedule(
                self.classes, self.rooms, self.teachers, self.groups)
        else:
            self.alpha = alpha  # Room for i-th class
            self.tau = tau  # Period for i-th class

    def conflicts(self, class_num_i, class_num_j):
        num_of_conflicts = 0
        if (
            self.alpha[class_num_i] == self.alpha[class_num_j]
            and self.tau[class_num_i] == self.tau[class_num_j]
        ):
            if (
                self.classes[class_num_i]["discipline"]
                != self.classes[class_num_j]["discipline"]
                or self.alpha[class_num_i]["room_type"] != RoomType.B
                or self.classes[class_num_i]["class_type"] != ClassType.LECTURE
            ):
                # print(
                #     'Два занятия в одной аудитории в одно и то же время')
                num_of_conflicts += 1
        # check course for one teacher in same time
        if (
            self.classes[class_num_i]["teacher"] == self.classes[class_num_j]["teacher"]
            and self.tau[class_num_i] == self.tau[class_num_j]
        ):
            # print('У препода два занятия в одно время')
            num_of_conflicts += 1
        # check same group for one class in same time
        if (
            self.classes[class_num_i]["group"] == self.classes[class_num_j]["group"]
            and self.tau[class_num_i] == self.tau[class_num_j]
        ):
            # print(
            #     'Одна группа на разных занятиях в одно время')
            num_of_conflicts += 1
        return num_of_conflicts

    def fitness(self):
        conflict = 0
        quality = 1e-6
        # list_week = [[], [], [], [], [], []]
        # for idx, period in enumerate(self.tau):
        #     if
        #     list_week[period['weekday']].append(self.classes[idx])
        # for day in list_week:
        #     if len(day) in (3, 4):
        #         quality += 2

        for class_num_i in range(self.number_of_classes - 1):
            for class_num_j in range(class_num_i + 1, self.number_of_classes):
                # conflict
                # check course in same time and same room
                conflict += self.conflicts(class_num_i, class_num_j)

                # quality
                # check group on the same day not in the same time
                if (
                    self.classes[class_num_i]["group"]
                    == self.classes[class_num_j]["group"]
                    and self.tau[class_num_i]["weekday"]
                    == self.tau[class_num_j]["weekday"]
                    and self.tau[class_num_i]["time_period"]
                    != self.tau[class_num_j]["time_period"]
                ):
                    quality += 1
                # check teacher on the same day not in the same time
                if (
                    self.classes[class_num_i]["teacher"]
                    == self.classes[class_num_j]["teacher"]
                    and self.tau[class_num_i]["weekday"]
                    == self.tau[class_num_j]["weekday"]
                    and self.tau[class_num_i]["time_period"]
                    != self.tau[class_num_j]["time_period"]
                ):
                    quality += 1
                # check group has same room on adjasent time periods
                if (
                    self.classes[class_num_i]["group"]
                    == self.classes[class_num_j]["group"]
                    and self.tau[class_num_i]["weekday"]
                    == self.tau[class_num_j]["weekday"]
                    and np.abs(
                        self.tau[class_num_i]["time_period"]
                        - self.tau[class_num_j]["time_period"]
                    )
                    == 1
                ):
                    quality += 1
            # check lecture in early periods
            if self.classes[class_num_i][
                "class_type"
            ] == ClassType.LECTURE and self.tau[class_num_i]["time_period"] in (
                1,
                2,
                3,
            ):
                quality += 1
            # check practical or laboratory in early periods
            if (
                self.classes[class_num_i]["class_type"] == ClassType.PRACTICAL
                or self.classes[class_num_i]["class_type"] == ClassType.LABORATORY
            ) and self.tau[class_num_i]["time_period"] in (4, 5, 6, 7):
                quality += 1

        if conflict == 0:
            schedule_group = {}
            schedule_teacher = {}
            for idx, class_i in enumerate(self.classes):
                group = self.groups[class_i['group']]
                teacher = self.teachers[class_i['teacher']]
                g_schedule = schedule_group.get(group, [False] * (6*7))
                t_schedule = schedule_teacher.get(teacher, [False] * (6*7))
                period = self.tau[idx]
                day = period['weekday'].value - 1
                time = period['time_period']
                g_schedule[day * 7 +
                           time] = True
                t_schedule[day * 7 +
                           time] = True
                schedule_group[group] = g_schedule
                schedule_teacher[teacher] = t_schedule

            for schedule_i in schedule_group.values():
                for day in range(6):
                    amount = 0
                    gaps = 0
                    all_gaps = 0
                    previous = False
                    for time in range(7):
                        if schedule_i[day * 7 + time]:
                            amount += 1
                            previous = True
                            if gaps:
                                all_gaps += gaps
                                gaps = 0
                        elif previous:
                            gaps += 1
                    if amount in (3, 4):
                        quality += 1
                    if all_gaps in (0, 1):
                        quality += 1
            for schedule_i in schedule_teacher.values():
                for day in range(6):
                    amount = 0
                    gaps = 0
                    all_gaps = 0
                    previous = False
                    for time in range(7):
                        if schedule_i[day * 7 + time]:
                            amount += 1
                            previous = True
                            if gaps:
                                all_gaps += gaps
                                gaps = 0
                        elif previous:
                            gaps += 1
                        if day == 6 and time in np.arange(3, 7):
                            quality -= 1
                    if amount in (3, 4):
                        quality += 1
                    if all_gaps in (0, 1):
                        quality += 1

        return -conflict, quality

    def breed(self, other: "Schedule"):
        indicies = np.random.choice(self.number_of_classes, 4, replace=False)
        new_alpha = self.alpha.copy()
        new_alpha[indicies] = other.alpha[indicies]
        new_tau = self.tau.copy()
        new_tau[indicies] = other.tau[indicies]
        return Schedule(
            self.classes,
            self.rooms,
            self.teachers,
            self.groups,
            self.number_of_classes,
            alpha=new_alpha,
            tau=new_tau,
        )

    def mutate(self) -> None:
        indicies = np.random.choice(
            self.number_of_classes,
            self.number_of_genes_to_mutate,
            replace=False,
        )
        room_indicies_to_change = np.random.choice(
            len(self.rooms),
            self.number_of_genes_to_mutate,
            replace=False,
        )
        # print(room_indicies_to_change, type(room_indicies_to_change))
        time_indicies_to_change = np.random.choice(
            len(self.t),
            self.number_of_genes_to_mutate,
            replace=False,
        )
        # print(time_indicies_to_change, type(time_indicies_to_change))
        self.alpha[indicies] = np.array(self.rooms)[room_indicies_to_change]
        self.tau[indicies] = np.array(self.t)[time_indicies_to_change]
        return

    def copy(self) -> Chromosome:
        return Schedule(
            self.classes,
            self.rooms,
            self.teachers,
            self.groups,
            number_of_classes=self.number_of_classes,
            alpha=self.alpha.copy(),
            tau=self.tau.copy(),
            number_of_genes_to_mutate=self.number_of_genes_to_mutate,
        )
