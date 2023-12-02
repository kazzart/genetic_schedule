from abc import ABC, abstractmethod
from copy import copy
from joblib import cpu_count, Parallel, delayed
import random
import numpy as np


class Chromosome(ABC):
    """
    Abstract base class that defines the interface for a chromosome in a genetic algorithm.

    Methods:
        - fitness(**kwargs) -> tuple[float, float]: Calculates and returns the fitness of the chromosome.
        - breed(other: "Chromosome") -> "Chromosome": Breeds the chromosome with another chromosome and returns a new chromosome.
        - mutate() -> None: Mutates the chromosome.
        - copy() -> "Chromosome": Creates and returns a copy of the chromosome.
    """

    @abstractmethod
    def fitness(self, **kwargs) -> tuple[float, float]:
        """
        Calculates and returns the fitness of the chromosome.

        Args:
            - **kwargs: Additional keyword arguments that may be needed for fitness calculation.

        Returns:
            - tuple[float, float]: A tuple of two floats representing the fitness values.
        """

    @abstractmethod
    def breed(self, other: "Chromosome") -> "Chromosome":
        """
        Breeds the chromosome with another chromosome and returns a new chromosome.

        Args:
            - other: Another Chromosome object to breed with.

        Returns:
            - Chromosome: A new Chromosome object resulting from the breeding process.
        """

    @abstractmethod
    def mutate(self) -> None:
        """
        Mutates the chromosome.
        """

    @abstractmethod
    def copy(self) -> "Chromosome":
        """
        Creates and returns a copy of the chromosome.

        Returns:
            - Chromosome: A new Chromosome object that is a copy of the original chromosome.
        """


class GeneticAlgorithm:
    """The `GeneticAlgorithm` class is a implementation of a genetic algorithm, which is a search algorithm inspired by the process of natural selection. It is used to find approximate solutions to optimization and search problems.

        Example Usage:
            # Create a chromosome class that inherits from the abstract base class `ABC`
            class Chromosome(ABC):
                @abstractmethod
                def fitness(self, **kwargs):

                @abstractmethod
                def breed(self, other):

                @abstractmethod
                def mutate(self):

            # Create an instance of the `GeneticAlgorithm` class
            ga = GeneticAlgorithm(
                Chromosome=Chromosome,
                population_size=100,
                elite_portion=0.1,
                init_args={},
                fitness_args={},
                max_iter=100,
                max_iter_no_improve=None,
                n_jobs=None
            )

            # Run the genetic algorithm
            best_solution = ga.run()

        In this example, we create a chromosome class that inherits from the abstract base class `ABC`. We then create an instance of the `GeneticAlgorithm` class, specifying the chromosome class, population size, elite portion, initialization arguments, fitness arguments, maximum number of iterations, maximum number of iterations without improvement, and number of parallel jobs. We then run the genetic algorithm using the `run` method, which returns the best solution found.

        Main functionalities:
            - Initialization of the genetic algorithm with the specified parameters.
            - Generation of an initial population of chromosomes.
            - Evaluation of the fitness of each chromosome in the population.
            - Ranking of the chromosomes based on their fitness.
            - Breeding of new chromosomes through crossover of the elite chromosomes.
            - Mutation of non-elite chromosomes.
            - Selection of the best solution based on fitness.
            - Iteration of the genetic algorithm until a stopping criterion is met.

        Methods:
            - `__init__(self, Chromosome, population_size, elite_portion, init_args, fitness_args, max_iter=None, max_iter_no_improve=None, n_jobs=None)`: Initializes the genetic algorithm with the specified parameters.
            - `_get_fitnesses(self)`: Evaluates the fitness of each chromosome in the population.
            - `_rank(self)`: Ranks the chromosomes based on their fitness.
            - `_breed(self)`: Breeds new chromosomes through crossover of the elite chromosomes.
            - `_mutate(self)`: Mutates non-elite chromosomes.
            - `next_generation(self)`: Generates the next generation of chromosomes.
            - `run(self) -> Chromosome`: Runs the genetic algorithm until a stopping criterion is met and returns the best solution found.

        Fields:
            - `best_solution: Chromosome`: The best solution found so far.
            - `history: list[tuple[float, float]]`: A list of tuples representing the fitness history of the best solution. Each tuple contains the fitness values for the x and y dimensions.
    """
    best_solution: Chromosome
    history: 'list[tuple[float, float]]'

    def __init__(
        self,
        Chromosome: Chromosome,
        population_size: int,
        elite_portion: int,
        init_args: dict,
        fitness_args: dict,
        max_iter: int | None = 100,
        max_iter_no_improve: int | None = None,
        n_jobs: int | None = None,
    ):
        self.population_size = population_size
        self.elite_size = int(self.population_size * elite_portion)
        self.breed_len = self.population_size - self.elite_size
        self.fitness_args = fitness_args
        self.max_iter = max_iter
        self.max_iter_no_improve = max_iter_no_improve
        self.iter = 0
        self.iter_no_improve = 0
        self.n_jobs = n_jobs or cpu_count()

        self.population = np.asarray(
            [
                Chromosome(**init_args) for _ in range(population_size)  # type: ignore
            ]
        )
        self.fitnesses = np.empty(
            shape=(self.population_size), dtype=np.float32)
        self._get_fitnesses()
        self._rank()
        if self.fitnesses[-1][0] < 0:
            print('Есть конфликты')
        self.best_solution = self.population[0].copy()
        self.history = [self.fitnesses[0]]

    def _get_fitnesses(self):
        self.fitnesses = np.array(
            [chromosome.fitness(**self.fitness_args)
             for chromosome in self.population], dtype=[('x', 'f4'), ('y', 'f4')]
        )

    def _rank(self):
        indicies = np.flip(np.argsort(self.fitnesses, order=('x', 'y')))
        self.population = self.population[indicies]
        self.fitnesses = self.fitnesses[indicies]

    def _breed(self):

        weights = np.divide(np.arange(self.population_size),
                            (self.population_size - 1) * self.population_size / 2)
        weights = np.flip(weights)
        offsprings = self.population[: self.elite_size].tolist()
        indicies = np.arange(self.population_size)
        breed1 = np.random.choice(indicies, self.breed_len, p=weights)
        breed1 = self.population[breed1]
        breed2 = np.random.choice(indicies, self.breed_len, p=weights)
        breed2 = self.population[breed2]
        for i in np.arange(self.breed_len):
            offsprings.append(breed1[i].breed(breed2[i]))  # type: ignore
        self.population = np.asarray(offsprings)

    def _mutate(self):
        for i in np.arange(self.elite_size, self.population_size):
            self.population[i].mutate()  # type: ignore

    def next_generation(self):
        self._breed()
        self._mutate()
        self._get_fitnesses()
        self._rank()
        self.history.append(self.fitnesses[0])
        if self.best_solution.fitness(**self.fitness_args)[0] < self.history[-1][0]:
            self.best_solution = self.population[0].copy()
            self.iter_no_improve = 0
        elif self.best_solution.fitness(**self.fitness_args)[0] == self.history[-1][0] and self.best_solution.fitness(**self.fitness_args)[1] < self.history[-1][1]:
            self.best_solution = self.population[0].copy()
            self.iter_no_improve = 0
        else:
            self.iter_no_improve += 1
        self.iter += 1

    def run(self) -> Chromosome:
        while (self.max_iter is None or self.iter < self.max_iter) and (
            self.max_iter_no_improve is None
            or self.iter_no_improve < self.max_iter_no_improve
        ):
            self.next_generation()
            if self.iter % 10 == 0:
                print(f'{self.iter} iteration')
        return self.best_solution
