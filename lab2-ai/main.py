import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import pandas as pd
from skfuzzy import control as ctrl
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

x = np.arange(0, 6, 0.1)
y = np.sin(x) + np.cos(x) * np.sin(np.cos(x))
z = np.sin(5 * np.cos(x) / 2) + np.cos(y)

plt.title('Plot for y-function')
plt.plot(x, y)
plt.grid(True)
plt.show()

plt.title('Plot for z-function')
plt.plot(x, z)
plt.grid(True)
plt.show()

x_antecedents = ctrl.Antecedent(x, 'x')
y_antecedents = ctrl.Antecedent(np.arange(min(y), max(y), 0.1), 'y')
z_consequents = ctrl.Consequent(np.arange(min(z), max(z), 0.1), 'z')


def calculate_intervals(values, n_intervals):
    start = min(values)
    end = max(values)
    intervals = []
    val = np.linspace(start, end, n_intervals + 1)
    for i in range(n_intervals):
        interval_start = val[i]
        interval_end = val[i + 1]
        mid = (interval_end + interval_start) / 2
        intervals.append([interval_start, mid, interval_end])
    diff = (intervals[0][1] - intervals[0][0]) / 2
    return intervals, diff


x_intervals, x_diff = calculate_intervals(x, 6)
y_intervals, y_diff = calculate_intervals(y, 6)
z_intervals, z_diff = calculate_intervals(z, 9)


def create_trianglemf(variable, intervals, names):
    for i, (interval, name) in enumerate(zip(intervals, names)):
        universe = variable.universe
        a = interval[0] - (interval[1] - interval[0]) * 0.15
        b = interval[1]
        c = interval[2] + (interval[2] - interval[1]) * 0.15
        variable[name] = fuzz.trimf(universe, [a, b, c])


def create_gaussmf(variable, intervals, names):
    for i, (interval, name) in enumerate(zip(intervals, names)):
        universe = variable.universe
        variable[name] = fuzz.gaussmf(universe, interval[1], 0.1)


x_names = ['mx1', 'mx2', 'mx3', 'mx4', 'mx5', 'mx6']
y_names = ['my1', 'my2', 'my3', 'my4', 'my5', 'my6']
z_names = ['mf1', 'mf2', 'mf3', 'mf4', 'mf5', 'mf6', 'mf7', 'mf8', 'mf9']

create_gaussmf(x_antecedents, x_intervals, x_names)
create_gaussmf(y_antecedents, y_intervals, y_names)
create_gaussmf(z_consequents, z_intervals, z_names)

x_antecedents.view()

y_antecedents.view()

z_consequents.view()

labels_x = x_antecedents.terms.keys()
labels_y = y_antecedents.terms.keys()
universe = np.arange(-6, 6, 0.1)
table_values = pd.DataFrame(index=labels_y, columns=labels_x)
table_names = pd.DataFrame(index=labels_y, columns=labels_x)

for label_x in labels_x:
    for label_y in labels_y:
        membership_x = fuzz.interp_membership(x_antecedents.universe, x_antecedents[label_x].mf, universe)
        membership_y = fuzz.interp_membership(y_antecedents.universe, y_antecedents[label_y].mf, universe)
        max_x = universe[np.argmax(membership_x)]
        max_y = universe[np.argmax(membership_y)]
        value_at_maximum = np.sin(5 * np.cos(max_x) / 2) + np.cos(max_y)
        value_at_maximum = min(max(z_consequents.universe), max(min(z_consequents.universe), value_at_maximum))
        table_values.at[label_y, label_x] = value_at_maximum

        for i, (min_range, _, max_range) in enumerate(z_intervals, 1):
            if min_range <= value_at_maximum <= max_range:
                table_names.at[label_y, label_x] = 'mf' + str(i)
                break

print('\nTable of function values at the maximum membership points:')
print(table_values)
print('\nTable of function names:')
print(table_names)

rules = []

for col_name in table_names.columns:
    for row_name in table_names.index:
        label_z = table_names.at[row_name, col_name]
        antecedent = (x_antecedents[col_name] & y_antecedents[row_name])
        consequent = z_consequents[label_z]
        rules.append(ctrl.Rule(antecedent=antecedent, consequent=consequent))
        print('if (x is ' + str(col_name) + ') and (y is ' + str(row_name) + ') then (f is ' + str(label_z) + ')')

system = ctrl.ControlSystem(rules)

simulation = ctrl.ControlSystemSimulation(system)
simulated = []
for x_val, y_val in zip(x, y):
    simulation.input['x'] = x_val
    simulation.input['y'] = y_val
    simulation.compute()
    simulated.append(simulation.output['z'])

plt.title('Comparison of z-function and simulation')
plt.plot(z, label='Z-function')
plt.legend()
plt.plot(simulated, label='Simulated')
plt.legend()
plt.xlabel('x')
plt.ylabel('z')
plt.show()

mae = mean_absolute_error(z, simulated)
mse = mean_squared_error(z, simulated)
r2 = r2_score(z, simulated)
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R2 Score: {r2}")