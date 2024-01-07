import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def prepare_data(file_name):
    df = pd.read_csv(file_name)
    features = df[['Open', 'High', 'Low']]
    target = df['Close']

    x = torch.tensor(features.values).float()
    y = torch.tensor(target.values).float()

    plt.figure(figsize=(10, 5))
    plt.plot(y)
    plt.title('Real values of ETH-USD')
    plt.xlabel('Data time')
    plt.ylabel('Value')
    plt.show()

    test_size = len(df) - 1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=False)

    return x_train, x_test, y_train.unsqueeze(1), y_test.unsqueeze(1), df


class HybridNeuralNetwork(nn.Module):
    def __init__(self):
        super(HybridNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to train the model
def train_model(model, x_train, y_train, x_test, y_test, epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    training_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        training_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            test_outputs = model(x_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())

            print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

    return training_losses, test_losses


def plot_losses(training_losses, test_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training loss')
    plt.plot(test_losses, label='Test loss')
    plt.title('Training and Test Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def predict_and_plot(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(x_test)
        plt.figure(figsize=(10, 5))
        plt.plot(y_test.numpy(), label='Real Value')
        plt.plot(predictions.numpy(), label='Predicted Value')
        plt.title('Comparison of Real and Predicted Values')
        plt.xlabel('Data time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, dataframe = prepare_data('ETH-USD.csv')
    neural_network = HybridNeuralNetwork()
    training_losses, test_losses = train_model(neural_network, x_train, y_train, x_test, y_test)
    plot_losses(training_losses, test_losses)
    predict_and_plot(neural_network, x_test, y_test)
