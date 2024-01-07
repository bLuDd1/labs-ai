import numpy as np
from random import random
from math import floor
from evol import Population, Evolution
from matplotlib import pyplot


def get_y(x):
    return np.sin(x) + np.cos(x) * np.sin(np.cos(x))


def get_z(x, y):
    return np.sin(5 * np.cos(x) / 2) + np.cos(y)


def evolve_population(chromosomes, eval_function, maximize, evolution_steps=20):
    evolution = Evolution()\
        .survive(fraction=0.01)\
        .breed(
            parent_picker=lambda arr: (arr[floor(random() * len(arr))], arr[floor(random() * len(arr))]),
            combiner=lambda a, b: tuple((a[i] + b[i]) / 2 for i in range(len(a)))
        )\
        .mutate(mutate_function=lambda xy: tuple(xy[i] + 0.001 * (random() - 0.5) for i in range(len(xy))))

    population = Population(
        chromosomes=chromosomes,
        eval_function=eval_function,
        maximize=maximize
    )

    result = population.evolve(evolution, evolution_steps).evaluate()
    return result


def main():
    minX = maxX = maxY = np.arange(-2, 5, 0.01)

    result_max = evolve_population(
        chromosomes=tuple((minX[i],) for i in range(len(minX))),
        eval_function=lambda x: get_y(*x),
        maximize=False
    )

    x_min = max(result_max, key=lambda x: x.fitness).chromosome[0]
    print(f'For the function y(x), the minimum is y({x_min}) = {get_y(x_min)}')

    result_min = evolve_population(
        chromosomes=tuple((maxX[i], maxY[i]) for i in range(len(maxY))),
        eval_function=lambda xy: get_z(*xy),
        maximize=True
    )

    x_max, y_max = max(result_min, key=lambda x: x.fitness).chromosome
    print(f'For the function z(x, y), the maximum is z({x_max}, {y_max}) = {get_z(x_max, y_max)}')

    pyplot.plot(minX, np.vectorize(get_y)(minX))
    pyplot.figure()
    plot = pyplot.axes(projection='3d')
    X, Y = np.meshgrid(maxX, maxY)
    plot.plot_surface(X, Y, np.vectorize(get_z)(X, Y))
    pyplot.show()


if __name__ == "__main__":
    main()
