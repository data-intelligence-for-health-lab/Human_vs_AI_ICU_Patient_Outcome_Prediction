import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
# from torch.utils.data import DataLoader, TensorDataset

# Define neural network classification model
class NeuralNetClassifier(nn.Module, BaseEstimator, ClassifierMixin):
    def __init__(self, input_size, neurons_per_layer, num_layers, learning_rate, dropout_rate, 
                activation_name, optimizer_name, epochs=10, batch_size=32):
        super().__init__()

        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Activation function mapping
        activation_function = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid
        }
        if activation_name not in activation_function:
            raise ValueError(f"Unsupported activation function: {activation_name}")
        self.activation = activation_function[activation_name]()  # Instantiate activation function

        # Optimizer mapping (initialized later in _build_model)
        self.optimizer_name = optimizer_name
        optimizer_function = {
            'adam': torch.optim.Adam,
            'rmsprop': torch.optim.RMSprop,
            'sgd': torch.optim.SGD
        }
        if optimizer_name not in optimizer_function:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        self.optimizer_class = optimizer_function[optimizer_name]  # Store class, not instance

        # Build model
        self._build_model()

    def _build_model(self):
        """Compiles model with specified architecture"""
        layers = []
        in_features = self.input_size

        # Loop through the number of layers and create each one
        for i in range(self.num_layers):
            out_features = self.neurons_per_layer[i]  # Get the number of neurons for this layer
            layers.append(nn.Linear(in_features, out_features))  # Fully connected layer
            layers.append(self.activation)  # Activation function
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))  # Dropout layer
            in_features = out_features  # Update in_features for next layer

        layers.append(nn.Linear(in_features, 1))  # Output layer with 1 neuron
        layers.append(nn.Sigmoid())  # Sigmoid activation for binary classification

        self.model = nn.Sequential(*layers).to(self.device)

        if torch.cuda.device_count() > 1:
            print(f'Distributing training across {torch.cuda.device_count()} GPUs')
            self.model = nn.DataParallel(self.model)

        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy loss

    def forward(self, x):
        """Forward pass"""
        return self.model(x)

    def fit(self, X, y):
        """Train the model"""
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        if isinstance(y, pd.Series):
            y = y.to_numpy()  # Convert Pandas Series to NumPy array
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        """Returns binary predictions"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X_tensor).cpu().numpy().flatten()
        return (outputs > 0.5).astype(int)

    def predict_proba(self, X):
        """Returns predicted probabilities"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X_tensor).cpu().numpy().flatten()
        return np.vstack((1 - outputs, outputs)).T  # Return probabilities in [P(0), P(1)] format
    
def make_final_neural_network(X_train, best_c_hyperparameters):
    """
    Initializes a NeuralNetClassifier with correctly formatted hyperparameters 
    and moves it to the correct device (GPU if available).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure input size is correctly defined
    best_c_hyperparameters["input_size"] = X_train.shape[1]
    
    # Extract and format neurons per layer
    num_layers = best_c_hyperparameters.get("num_layers")
    best_c_hyperparameters["neurons_per_layer"] = [
        best_c_hyperparameters.get(f"neurons_layer_{i}")
        for i in range(num_layers)
    ]
    
    # Remove individual neuron layer keys
    for i in range(num_layers):
        best_c_hyperparameters.pop(f"neurons_layer_{i}", None)

    # Move model to GPU
    return NeuralNetClassifier(**best_c_hyperparameters).to(device)

# Define neural network regression model
class NeuralNetRegressor(nn.Module, BaseEstimator, RegressorMixin):
    def __init__(self, input_size, neurons_per_layer, num_layers, learning_rate, dropout_rate, 
                 activation_name, optimizer_name, epochs=10, batch_size=32):
        super().__init__()

        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # Activation function mapping
        activation_function = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid
        }
        if activation_name not in activation_function:
            raise ValueError(f"Unsupported activation function: {activation_name}")
        self.activation = activation_function[activation_name]()  

        # Optimizer mapping
        optimizer_function = {
            'adam': optim.Adam,
            'rmsprop': optim.RMSprop,
            'sgd': optim.SGD
        }
        if optimizer_name not in optimizer_function:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        self.optimizer_class = optimizer_function[optimizer_name]  

        # Build model
        self._build_model()

    def _build_model(self):
        """Compiles model with specified architecture"""
        layers = []
        in_features = self.input_size

        for i in range(self.num_layers):
            out_features = self.neurons_per_layer[i]  
            layers.append(nn.Linear(in_features, out_features))  
            layers.append(self.activation)  
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))  
            in_features = out_features  

        layers.append(nn.Linear(in_features, 1))  # Output layer (no activation for regression)

        self.model = nn.Sequential(*layers).to(self.device)

        if torch.cuda.device_count() > 1:
            print(f'Distributing training across {torch.cuda.device_count()} GPUs')
            self.model = nn.DataParallel(self.model)

        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()  # Mean Squared Error loss for regression

    def forward(self, x):
        """Forward pass"""
        return self.model(x)

    def fit(self, X, y):
        """Train the model"""
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        """Returns predicted values"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X_tensor).cpu().numpy().flatten()
        return outputs