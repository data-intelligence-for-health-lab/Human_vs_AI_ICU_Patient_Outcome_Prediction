import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

# Define neural network classification model
class NeuralNetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, neurons_per_layer, num_layers, learning_rate, dropout_rate, 
                activation_name, optimizer_name, epochs, batch_size):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation_name = activation_name
        
        # Activation function mapping
        activation_function = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid
        }
        if activation_name not in activation_function:
            raise ValueError(f"Unsupported activation function: {activation_name}")
        self.activation = activation_function[self.activation_name]

        # Optimizer mapping (initialized later in _build_model)
        self.optimizer_name = optimizer_name
        optimizer_function = {
            'adam': torch.optim.Adam,
            'rmsprop': torch.optim.RMSprop,
            'sgd': torch.optim.SGD
        }
        if self.optimizer_name not in optimizer_function:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        self.optimizer_class = optimizer_function[self.optimizer_name]  # Store class, not instance

        # Build model
        #self._build_model()
        self.model = None
        
    def _build_model(self, input_size):
        """Compiles model with specified architecture"""
        layers = []
        in_features = input_size
        print("in_features", in_features)
        # Loop through the number of layers and create each one
        for i in range(self.num_layers):
            out_features = self.neurons_per_layer[i]  # Get the number of neurons for this layer
            layers.append(nn.Linear(in_features, out_features))  # Fully connected layer
            layers.append(self.activation())  # Activation function
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
        y_tensor = torch.tensor(y.to_numpy(dtype=np.float32)).view(-1, 1).to(self.device)

        if self.model is None:
            print("Building model before training...")
            self._build_model(input_size=X_tensor.shape[1])

        self.model.to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
    
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss/len(dataloader):.4f}")

    def predict(self, X):
        """Returns binary predictions"""
        self.model.eval()  
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X_tensor).cpu().numpy().flatten()
        return (outputs > 0.5).astype(int)

    def predict_proba(self, X):
            """Returns predicted probabilities"""
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                outputs = self.model(X_tensor).cpu().numpy().flatten()
            return np.vstack((1 - outputs, outputs)).T

def make_final_neural_network(best_c_hyperparameters):
    """
    Initializes a NeuralNetClassifier with correctly formatted hyperparameters 
    and moves it to the correct device (GPU if available).
    """
    # Extract and format neurons per layer
    num_layers = best_c_hyperparameters.get("num_layers", 0)

    # Create a new dictionary to avoid modifying the original
    formatted_params = best_c_hyperparameters.copy()
    
    # Extract and remove individual layer neuron counts
    formatted_params["neurons_per_layer"] = [
        formatted_params.pop(f"neurons_layer_{i}", None) for i in range(num_layers)
    ]

    # Ensure all None values are removed (if any)
    formatted_params["neurons_per_layer"] = [
        n for n in formatted_params["neurons_per_layer"] if n is not None
    ]

    # Ensure no unexpected keys remain
    for key in list(formatted_params.keys()):
        if key.startswith("neurons_layer_"):
            formatted_params.pop(key)

    # Move model to GPU
    return NeuralNetClassifier(**formatted_params)

# Define neural network regression model
class NeuralNetRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, neurons_per_layer, num_layers, learning_rate, dropout_rate, 
                 activation_name, optimizer_name, epochs=10, batch_size=32):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation_name = activation_name
        self.optimizer_name = optimizer_name
        self.model = None  # Model initialized in fit()

        # Activation function mapping
        activation_function = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid
        }
        if activation_name not in activation_function:
            raise ValueError(f"Unsupported activation function: {activation_name}")
        self.activation = activation_function[self.activation_name]

        # Optimizer mapping (initialized later in _build_model)
        optimizer_function = {
            'adam': optim.Adam,
            'rmsprop': optim.RMSprop,
            'sgd': optim.SGD
        }
        if self.optimizer_name not in optimizer_function:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        self.optimizer_class = optimizer_function[self.optimizer_name]  # Store class, not instance

    def _build_model(self, input_size):
        """Compiles model with specified architecture"""
        layers = []
        in_features = input_size
        
        for i in range(self.num_layers):
            out_features = self.neurons_per_layer[i]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(self.activation())  # Activation function
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            in_features = out_features

        layers.append(nn.Linear(in_features, 1))  # Output layer for regression

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
        y_tensor = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.float32).view(-1, 1).to(self.device)

        if self.model is None:
            print("Building model before training...")
            self._build_model(input_size=X_tensor.shape[1])

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss/len(dataloader):.4f}")

    def predict(self, X):
        """Returns predicted values"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X_tensor).cpu().numpy().flatten()
        return outputs