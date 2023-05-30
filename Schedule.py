import numpy as np
from GeneticAlgorithm import Chromosome
from enums import RoomType, Weekday, ClassType
from Generator import generate_schedule


class Schedule(Chromosome):
    def __init__(
        self,
        classes: list,
        rooms: list,
        number_of_classes: int,
        alpha: np.ndarray | None = None,
        tau: np.ndarray | None = None,
        number_of_genes_to_mutate: int = 1,
    ) -> None:
        # Solution vectors
        self.classes = classes
        self.rooms = rooms
        self.number_of_classes = number_of_classes
        self.number_of_genes_to_mutate = number_of_genes_to_mutate
        if alpha is None or tau is None:
            self.alpha, self.tau = generate_schedule(self.classes, self.rooms)
        else:
            self.alpha = alpha  # Room for i-th class
            self.tau = tau  # Period for i-th class

    def conflicts(self, class_num_i, class_num_j):
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
                return True
        # check course for one teacher in same time
        if (
            self.classes[class_num_i]["teacher"] == self.classes[class_num_j]["teacher"]
            and self.tau[class_num_i] == self.tau[class_num_j]
        ):
            return True
        # check same group for one class in same time
        if (
            self.classes[class_num_i]["group"] == self.classes[class_num_j]["group"]
            and self.tau[class_num_i] == self.tau[class_num_j]
        ):
            return True
        return False

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
                if self.conflicts(class_num_i, class_num_j):
                    return 1e-6

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

        return quality

    def breed(self, other: "Schedule"):
        indicies = np.random.choice(self.number_of_classes, 4, replace=False)
        new_alpha = self.alpha.copy()
        new_alpha[indicies] = other.alpha[indicies]
        new_tau = self.tau.copy()
        new_tau[indicies] = other.tau[indicies]
        return Schedule(
            self.classes,
            self.rooms,
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
        alpha, tau = generate_schedule(self.classes, self.rooms)
        self.alpha[indicies] = alpha[indicies]
        self.tau[indicies] = tau[indicies]
        return

    def copy(self) -> Chromosome:
        return Schedule(
            self.classes,
            self.rooms,
            number_of_classes=self.number_of_classes,
            alpha=self.alpha.copy(),
            tau=self.tau.copy(),
            number_of_genes_to_mutate=self.number_of_genes_to_mutate,
        )
