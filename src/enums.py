from enum import Enum


class RoomType(Enum):
    """
    Types of rooms described in one letter:
        - T - small room 1 group laboratory classes
        - S - small room 1 group practical classes
        - B - big room any number of groups
    """

    T = 1
    S = 2
    B = 4


class Weekday(Enum):
    """Weekday enumerate"""

    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5
    SATURDAY = 6


class ClassType(Enum):
    """
    Class type enumerate
    Lecture, practical or laboratory class
    """

    LECTURE = 1
    PRACTICAL = 2
    LABORATORY = 3
