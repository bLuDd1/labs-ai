import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Triangle membership function
x = np.linspace(0, 10, 100)
func = fuzz.trimf(x, [4, 5, 6])

plt.plot(x, func)
plt.title('Triangle membership function')
plt.xlabel('X')
plt.ylabel('Membership value')
plt.ylim(-0.1, 1.1)
plt.grid(True)
plt.show()

# Trapezoid membership function
x = np.linspace(0, 10, 100)
func = fuzz.trapmf(x, [4, 5, 6, 9])

plt.plot(x, func)
plt.title('Trapezoid membership function')
plt.xlabel('X')
plt.ylabel('Membership value')
plt.ylim(-0.1, 1.1)
plt.grid(True)
plt.show()

# Simple Gaussian membership function
x = np.linspace(-20, 20, 100)
func = fuzz.gaussmf(x, 5, 3)

plt.plot(x, func)
plt.title('Simple Gaussian membership function')
plt.xlabel('X')
plt.ylabel('Membership value')
plt.grid(True)
plt.show()

# Gaussian membership functions
x = np.linspace(-20, 20, 100)
func1 = fuzz.gauss2mf(x, 4, 5, 5, 3)
func2 = fuzz.gauss2mf(x, 3.5, 3.6, 4, 2)
func3 = fuzz.gauss2mf(x, 2, 6, 6, 4)

plt.plot(x, func1)
plt.plot(x, func2)
plt.plot(x, func3)
plt.title('Gaussian membership functions')
plt.xlabel('X')
plt.ylabel('Membership value')
plt.grid(True)
plt.show()

# Generalized bell shape membership function
x = np.linspace(-20, 20, 100)
func = fuzz.gbellmf(x, 5, 5, 5)

plt.plot(x, func)
plt.title('Generalized bell shape membership function')
plt.xlabel('X')
plt.ylabel('Membership value')
plt.grid(True)
plt.show()

# Sigmoid membership function
x = np.linspace(0, 10, 100)
func = fuzz.sigmf(x, 3, 8)

plt.plot(x, func)
plt.title('Sigmoid membership function')
plt.xlabel('X')
plt.ylabel('Membership value')
plt.grid(True)
plt.show()

# Diff of two sigmoid membership functions
x = np.linspace(0, 10, 100)
func = fuzz.dsigmf(x, 3, 5, 8, 10)

plt.plot(x, func)
plt.title('Diff of two sigmoid membership functions')
plt.xlabel('X')
plt.ylabel('Membership value')
plt.grid(True)
plt.show()

# Prod of two sigmoid membership functions
x = np.linspace(0, 10, 100)
func = fuzz.psigmf(x, 3, 5, 7, 8)

plt.plot(x, func)
plt.title('Prod of two sigmoid membership functions')
plt.xlabel('X')
plt.ylabel('Membership value')
plt.grid(True)
plt.show()

# Z-function
x = np.linspace(0, 10, 100)
func = fuzz.zmf(x, 4, 5)

plt.plot(x, func)
plt.title('Z-function')
plt.xlabel('X')
plt.ylabel('Membership value')
plt.grid(True)
plt.show()

# PI-function
x = np.linspace(0, 10, 100)
func = fuzz.pimf(x, 4, 5, 6, 9)

plt.plot(x, func)
plt.title('PI-function')
plt.xlabel('X')
plt.ylabel('Membership value')
plt.grid(True)
plt.show()

# S-function
x = np.linspace(0, 10, 100)
func = fuzz.smf(x, 6, 9)

plt.plot(x, func)
plt.title('S-function')
plt.xlabel('X')
plt.ylabel('Membership value')
plt.grid(True)
plt.show()

# Logical or min function
x = np.linspace(0, 10, 100)
func1 = fuzz.gaussmf(x, 4, 2)
func2 = fuzz.gaussmf(x, 6, 2)

minFunc = np.fmin(func1, func2)

plt.plot(x, func1, 'b', linestyle='dotted')
plt.plot(x, func2, 'b', linestyle='dotted')
plt.plot(x, minFunc, 'b')
plt.title('Logical or min function')
plt.xlabel('X')
plt.ylabel('Membership value')
plt.grid(True)
plt.show()

# Logical or max function
maxFunc = np.fmax(func1, func2)

plt.plot(x, func1, 'b', linestyle='dotted')
plt.plot(x, func2, 'b', linestyle='dotted')
plt.plot(x, maxFunc, 'b')
plt.title('Logical or max function')
plt.xlabel('X')
plt.ylabel('Membership value')
plt.grid(True)
plt.show()

# Interpretation for conjunctive operation
production = func1 * func2

plt.plot(x, func1, 'b', linestyle='dotted')
plt.plot(x, func2, 'b', linestyle='dotted')
plt.plot(x, production, 'b')
plt.title('Interpretation for conjunctive operation')
plt.xlabel('X')
plt.ylabel('Membership value')
plt.grid(True)
plt.show()

# Interpretation for disjunctive operation
algSum = func1 + func2 - func1 * func2

plt.plot(x, func1, 'b', linestyle='dotted')
plt.plot(x, func2, 'b', linestyle='dotted')
plt.plot(x, algSum, 'b')
plt.title('Interpretation for disjunctive operation')
plt.xlabel('X')
plt.ylabel('Membership value')
plt.grid(True)
plt.show()

# Difference
x = np.linspace(0, 10, 100)
func = fuzz.gaussmf(x, 5, 2)

plt.plot(x, func)
plt.plot(x, 1 - func, 'b', linestyle='dotted')
plt.title('Difference')
plt.xlabel('X')
plt.ylabel('Membership value')
plt.grid(True)
plt.show()
