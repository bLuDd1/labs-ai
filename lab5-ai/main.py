d_pattern = [1, 1, -1,
             1, -1, 1,
             1, 1, -1]

t_pattern = [1, 1, 1,
             -1, 1, -1,
             -1, 1, -1]

v_pattern = [1, -1, 1,
             1, -1, 1,
             -1, 1, -1]

l_pattern = [1, -1, -1,
             1, -1, -1,
             1, 1, 1]

expected_result = [[1, -1, -1, -1],
                   [-1, 1, -1, -1],
                   [-1, -1, 1, -1],
                   [-1, -1, -1, 1]]

training_patterns = [d_pattern, t_pattern, v_pattern, l_pattern]
num_neurons = len(training_patterns)

d_mistake = [1, 1, -1,
             -1, -1, 1,
             1, 1, -1]

t_mistake = [1, 1, 1,
             -1, 1, -1,
             -1, -1, -1]

pattern_mistake = [d_pattern, t_pattern, v_pattern, l_pattern, d_mistake, t_mistake]


def add_bias_to_patterns(letters, neurons):
    result = []
    for pattern in letters[:neurons]:
        result.append([1] + pattern)
    return result


def initialize_weights_matrix(letters, neurons):
    result = []
    for _ in range(neurons):
        result.append([0] * len(letters[0]))
    return result


def update_weights(weights, letters, expected_result):
    for j in range(len(weights)):
        for i in range(len(weights[0])):
            for k in range(len(letters)):
                weights[j][i] += letters[k][i] * expected_result[j][k]
    return weights


def calculate_neuron_outputs(letters, weights, neurons):
    outputs = []

    for j in range(len(letters)):
        current_output = []
        for i in range(neurons):
            activation = sum(w * l for w, l in zip(weights[i], letters[j]))
            current_output.append(1 if activation > 0 else -1)

        outputs.append(current_output)

    return outputs


def hebb_network(letters, expected_result, neurons):
    letters_with_bias = add_bias_to_patterns(letters, neurons)
    weights = initialize_weights_matrix(letters_with_bias, neurons)

    weights = update_weights(weights, letters_with_bias, expected_result)

    actual_result = calculate_neuron_outputs(letters_with_bias, weights, neurons)

    if actual_result == expected_result:
        return weights

    raise Exception('Weight adaptation problem. Weights: ' + str(weights))


final_weights = hebb_network(training_patterns, expected_result, num_neurons)

print("Ваги:")
for weights in final_weights:
    print(weights)

for i in range(len(pattern_mistake)):
    pattern_mistake[i] = [1] + pattern_mistake[i]


actual_result = calculate_neuron_outputs(pattern_mistake, final_weights, num_neurons)

letter_names = ['D', 'T', 'V', 'L', 'D з помилкою', 'T з помилкою']
print('\nРезультат:\n D   T   V   L')
for i in range(len(actual_result)):
    print(actual_result[i], letter_names[i])