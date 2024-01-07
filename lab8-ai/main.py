from dataclasses import dataclass
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math


@dataclass
class Individual:
    chromosome: np.ndarray
    fitness: float


class GeneticAlgorithm:
    def __init__(self, elements, matrix, dimensions, individuals=10, generations=500, mutation_rate=0.01):
        self.matrix = matrix
        self.n = len(elements)
        self.dimensions = dimensions
        self.individuals_n = individuals
        self.gen_n = generations
        self.elements = elements
        self.mutation_rate = mutation_rate
        self.lmax = math.pow(self.n, 2) * math.sqrt(math.pow(dimensions[0], 2) + math.pow(dimensions[1], 2))
        self.nab = self.n * dimensions[0] * dimensions[1]
        self.current_generation = self.initialize_first_generation()

    def initialize_first_generation(self):
        individuals = []
        max_x, max_y = self.dimensions
        for _ in range(self.individuals_n):
            chromosome = [[np.random.uniform(element_dim/2, max_dim - element_dim/2) for max_dim, element_dim in zip(self.dimensions, element)] for element in self.elements]
            individual = Individual(chromosome=np.array(chromosome), fitness=0)
            individuals.append(individual)
        return np.array(individuals)

    def calculate_fitness(self):
        for individual in self.current_generation:
            connections = self.total_connections_length(individual)
            area = self.total_overlap_area(individual)
            fitness = self.mutation_rate * connections + area
            individual.fitness = round(fitness, 5)
            if area == 0:
                return True, individual
        return False, None

    def total_connections_length(self, individual):
        total_len = 0
        matrix = np.tril(self.matrix)
        for i, coord_a in enumerate(individual.chromosome):
            for j, coord_b in enumerate(individual.chromosome):
                if matrix[i, j] == 0:
                    continue
                total_len += math.sqrt(
                    math.pow(coord_a[0] - coord_b[0], 2) +
                    math.pow(coord_a[1] - coord_b[1], 2)
                ) * matrix[i, j]
        return total_len / self.lmax

    def total_overlap_area(self, individual):
        total_area = 0
        for i, coord_a in enumerate(individual.chromosome):
            x1, y1 = coord_a
            a1, b1 = self.elements[i]
            for j, coord_b in enumerate(individual.chromosome):
                if i == j:
                    continue
                a2, b2 = self.elements[j]
                x2, y2 = coord_b
                r1 = [x1 - a1 / 2, y1 - b1 / 2, x1 + a1 / 2, y1 + b1 / 2]
                r2 = [x2 - a2 / 2, y2 - b2 / 2, x2 + a2 / 2, y2 + b2 / 2]
                if r1[0] >= r2[2] or r1[2] <= r2[0] or r1[3] <= r2[1] or r1[1] >= r2[3]:
                    continue
                total_area += (0.5 * (a2 + a1) - abs(x2 - x1)) * (0.5 * (b2 + b1) - abs(y2 - y2))
        return total_area / self.nab

    def mutate(self, individual):
        i = random.choice(range(len(individual.chromosome)))
        coord = random.choice(range(2))
        old = individual.chromosome[i][coord]
        margin = self.elements[i][coord] / 2
        upper_bound = self.dimensions[coord] - margin
        while True:
            new = old * np.random.uniform(0.2, 2)
            if margin <= new <= upper_bound:
                break
        individual.chromosome[i][coord] = round(new, 2)

    def selection(self):
        sorting_key = lambda x: x.fitness
        gen_sorted = sorted(self.current_generation, key=sorting_key)
        idx = math.ceil(len(self.current_generation) / 2) + 1
        return gen_sorted[:idx]

    def crossover(self, individuals):
        father, mother = np.random.choice(individuals, 2, False)
        crossover_point = random.choice(range(1, self.n))
        father_tail = father.chromosome[:crossover_point]
        father_head = father.chromosome[crossover_point:]
        mother_tail = mother.chromosome[:crossover_point]
        mother_head = mother.chromosome[crossover_point:]
        first_child = Individual(chromosome=np.concatenate((father_tail, mother_head), axis=0), fitness=0)
        second_child = Individual(chromosome=np.concatenate((mother_tail, father_head), axis=0), fitness=0)
        return first_child, second_child

    def new_generation(self):
        best_fit = self.selection()
        new_gen = []
        while True:
            children = self.crossover(best_fit)
            for child in children:
                if len(new_gen) == self.n:
                    return np.array(new_gen)
                if np.random.uniform(0, 100) < 30:
                    self.mutate(child)
                new_gen.append(child)

    def run_genetic_algorithm(self):
        sorting_key = lambda x: x.fitness
        mean_history = list()
        min_history = list()
        for _ in range(self.gen_n):
            overlap_area_zero, individual = self.calculate_fitness()
            fit_list = [i.fitness for i in self.current_generation]
            mean_history.append(np.mean(fit_list))
            min_history.append(np.min(fit_list))
            if overlap_area_zero:
                return individual, mean_history, min_history
            self.current_generation = self.new_generation()
            self.calculate_fitness()
        best_fit = sorted(self.current_generation, key=sorting_key, reverse=True)[0]
        return best_fit, mean_history, min_history


def generate_elements(width, height, elements_n):
    total_area = width * height
    avg_area = total_area / elements_n
    while True:
        width_bound = min(1.3 * math.sqrt(avg_area), width)
        height_bound = min(1.3 * math.sqrt(avg_area), height)
        widths = np.random.uniform(0.5, width_bound, elements_n)
        heights = np.random.uniform(0.5, height_bound, elements_n)
        if np.dot(widths, heights) < total_area:
            return np.array((widths.round(2), heights.round(2))).T


def generate_matrix(elements_n):
    base = np.random.randint(0, 2, (elements_n, elements_n))
    result = np.tril(base) + np.tril(base, -1).T
    np.fill_diagonal(result, 0)
    return result


def plot_results(result, min_fit, mean_fit, elements, width, height):
    plt.figure()
    plt.plot(min_fit, label='min fitness')
    plt.plot(mean_fit, label='avg fitness')
    plt.legend()
    _, ax = plt.subplots()
    for i, (width_i, height_i) in enumerate(elements):
        x_center, y_center = result.chromosome[i]
        x = x_center - width_i / 2
        y = y_center - height_i / 2
        ax.add_patch(Rectangle((x, y), width_i, height_i, linewidth=1, edgecolor='black'))
        ax.text(x_center, y_center, str(i + 1))
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
    plt.show()


WIDTH = 60
HEIGHT = 60
ELEMENTS_N = 15
INDIVIDUALS = 300
GENERATIONS = 1000

elements = generate_elements(WIDTH, HEIGHT, ELEMENTS_N)
matrix = generate_matrix(ELEMENTS_N)
genetic_algorithm = GeneticAlgorithm(elements, matrix, (WIDTH, HEIGHT), INDIVIDUALS, GENERATIONS)
results = genetic_algorithm.run_genetic_algorithm()
plot_results(*results, elements, WIDTH, HEIGHT)

