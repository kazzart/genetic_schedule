from abc import ABC, abstractmethod
from copy import copy
from joblib import cpu_count, Parallel, delayed
import random
import numpy as np


class Chromosome(ABC):
    @abstractmethod
    def fitness(self, **kwargs) -> tuple[float, float]:
        pass

    @abstractmethod
    def breed(self, other: "Chromosome") -> "Chromosome":
        pass

    @abstractmethod
    def mutate(self) -> None:
        pass

    @abstractmethod
    def copy(self) -> "Chromosome":
        pass


@delayed
def breed_worker(parents1: np.ndarray, parents2: np.ndarray):
    offsprings = []
    for i in range(parents1.size):
        offsprings.append(parents1[i].breed(parents2[i]))
    return np.asarray(offsprings)


@delayed
def calc_fitness_worker(chromosomes: np.ndarray, args: dict):
    fitnesses = np.empty(shape=(chromosomes.shape[0]), dtype=np.float32)
    for i in np.arange(chromosomes.shape[0]):
        fitnesses[i] = chromosomes[i].fitness(**args)  # type: ignore
    return chromosomes, fitnesses


@delayed
def mutate_worker(chromosomes: np.ndarray):
    for i in range(chromosomes.size):
        chromosomes[i].mutate()
    return chromosomes


class GeneticAlgorithm:
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
                # type: ignore
                Chromosome(**init_args) for _ in range(population_size)
            ]  # type: ignore
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
        # parr = Parallel(self.n_jobs)
        # chunks = np.array_split(self.population, self.n_jobs)
        # res = parr(calc_fitness_worker(chunks[i], self.fitness_args) for i in range(self.n_jobs))
        # self.population, self.fitnesses = np.hstack(res)
        # self.fitnesses = self.fitnesses.astype(np.float32)
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
        # weights = np.divide(self.fitnesses, np.sum(self.fitnesses))
        # offsprings = []
        # for i in np.arange(self.elite_size):
        #     offsprings.append(self.population[i].copy())
        offsprings = self.population[: self.elite_size].tolist()
        indicies = np.arange(self.population_size)
        breed1 = np.random.choice(indicies, self.breed_len, p=weights)
        breed1 = self.population[breed1]
        breed2 = np.random.choice(indicies, self.breed_len, p=weights)
        breed2 = self.population[breed2]
        # parr = Parallel(self.n_jobs)
        # chunks1 = np.array_split(breed1, self.n_jobs)
        # chunks2 = np.array_split(breed2, self.n_jobs)
        # res = parr(breed_worker(chunks1[i], chunks2[i])
        #            for i in range(self.n_jobs))
        # for chunk in res:
        #     for obj in chunk:
        #         offsprings.append(obj.copy())
        for i in np.arange(self.breed_len):
            offsprings.append(breed1[i].breed(breed2[i]))  # type: ignore
        self.population = np.asarray(offsprings)

    def _mutate(self):
        # parr = Parallel(self.n_jobs)
        # chunks = np.array_split(self.population, self.n_jobs)
        # res = parr(mutate_worker(chunks[i]) for i in range(self.n_jobs))
        # self.population = np.hstack(res)
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
