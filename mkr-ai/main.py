from matplotlib import pyplot as plt
import numpy as np
import random


def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def target_function(x, y):
    return np.cos(y) + np.sin(x) + np.cos(x + y)


class NeuralNetwork:
    def __init__(self):
        self.weights = [
            np.random.rand(2, 3),
            np.random.rand(3, 6),
            np.random.rand(6, 8),
            np.random.rand(8, 6),
            np.random.rand(6, 3),
            np.random.rand(3, 1)
        ]

    def forward_propagate(self, inputs):
        layers = [inputs]
        for i in range(len(self.weights)):
            layers.append(sigmoid_activation(np.dot(layers[i], self.weights[i])))
        return layers[-1]

    def calculate_error(self, target):
        return np.mean(np.square(self.output - target))


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [NeuralNetwork() for _ in range(population_size)]

    def calculate_fitness(self, x, y, z):
        fitness_scores = []
        for network in self.population:
            output = network.forward_propagate(np.array([x, y]))
            error = np.mean(np.square(output - z))
            fitness_scores.append(1 / (error + 1e-6))
        return fitness_scores

    def select_parents(self, fitness_scores):
        return random.choices(self.population, weights=fitness_scores, k=2)

    def crossover(self, parent1, parent2):
        child = NeuralNetwork()
        for i in range(len(child.weights)):
            crossover_point = random.randint(0, len(parent1.weights[i]))
            child.weights[i][:crossover_point] = parent1.weights[i][:crossover_point]
            child.weights[i][crossover_point:] = parent2.weights[i][crossover_point:]
        return child

    def mutate(self, network):
        for i in range(4):  # Mutate first 4 layers
            mutation_mask = np.random.rand(*network.weights[i].shape) < self.mutation_rate
            network.weights[i] += np.random.randn(*network.weights[i].shape) * mutation_mask * 0.1
        return network

    def run_generation(self, x, y, z):
        fitness_scores = self.calculate_fitness(x, y, z)
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = self.select_parents(fitness_scores)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        self.population = new_population
        return min(fitness_scores)


population_size = 15
mutation_rate = 0.01
generations = 150

genetic_algorithm = GeneticAlgorithm(population_size, mutation_rate)

x_train = np.linspace(2, 4, 21)
y_train = np.linspace(2, 4, 21)

for generation in range(generations):
    minFit = []
    for x in x_train:
        for y in y_train:
            z = target_function(x, y)
            minFit.append(genetic_algorithm.run_generation(x, y, z))
    if generation % 10 == 0:
        print(generation)
best_network = genetic_algorithm.population[0]
errors = []

for x in x_train:
    for y in y_train:
        z = target_function(x, y)
        prediction = best_network.forward_propagate(np.array([x, y]))
        error = np.mean(np.square(prediction - np.array([z])))
        errors.append(error)

print("Evaluation Results:")

for i, (x, y) in enumerate(zip(x_train, y_train)):
    print(f"Input: ({round(x, 1)}, {round(y, 1)}), Error: {errors[i] / 10}")

Z_nn = np.zeros((21, 21))
x_vals = np.linspace(2, 4, 21)
y_vals = np.linspace(2, 4, 21)
z_vals = np.array([[target_function(x, y) for x in x_vals] for y in y_vals])

for i, x in enumerate(x_vals):
    for j, y in enumerate(y_vals):
        prediction = best_network.forward_propagate(np.array([x, y]))
        Z_nn[i, j] = prediction[0]

fig = plt.figure(figsize=(12, 6))
X, Y = np.meshgrid(x_vals, y_vals)

ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X, Y, z_vals, cmap='inferno', edgecolor='none')
ax1.set_title('Target Function')
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X, Y, Z_nn, cmap='magma', edgecolor='none')
ax2.set_title('Neural Network Approximation')
ax2.set_xlabel('X axis')
ax2.set_ylabel('Y axis')
ax2.set_zlabel('Z axis')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

plt.show()

error_matrix = np.reshape(errors, (21, 21))
X, Y = np.meshgrid(x_vals, y_vals)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, error_matrix, cmap='inferno')
plt.colorbar(surf)
ax.set_title('Neural Network Error Distribution')
ax.set_xlabel('X values')
ax.set_ylabel('Y values')
ax.set_zlabel('Error')
plt.show()
