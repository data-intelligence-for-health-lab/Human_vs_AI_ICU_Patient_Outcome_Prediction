import torch
import torch.nn as nn
import torch.nn.functional as F

# Define neural network model
class NeuralNetClassifier(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, dropout_rate):
        
        self for evey hyperparameters
        
        self.input_dim = input_size
        self.device = torch device
        
        
        self._build_model()
        
        # super(NeuralNetClassifier, self).__init__()
        
        # self.input_layer = nn.Linear(input_size, hidden_units)
        # self.hidden_layers = nn.ModuleList()
        # for _ in range(num_layers - 1):
        #     self.hidden_layers.append(nn.Linear(hidden_units, hidden_units))
        
        # self.output_layer = nn.Linear(hidden_units, 1)
        # self.dropout = nn.Dropout(dropout_rate)

    def fit(self, X, y):
        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device).view(-1, 1).to(self.device)
        dataload
    def _build_model(self, X, y):
    
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)  # No Sigmoid here if using BCEWithLogitsLoss
        return x
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # Define neural network model
# class NeuralNetClassifier(nn.Module):
#     def __init__(self, input_size, hidden_units, num_layers, dropout_rate):
#         super(NeuralNetClassifier, self).__init__()
        
#         self.input_layer = nn.Linear(input_size, hidden_units)
#         self.hidden_layers = nn.ModuleList()
#         for _ in range(num_layers - 1):
#             self.hidden_layers.append(nn.Linear(hidden_units, hidden_units))
        
#         self.output_layer = nn.Linear(hidden_units, 1)
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x):
#         x = F.relu(self.input_layer(x))
#         x = self.dropout(x)
#         for layer in self.hidden_layers:
#             x = F.relu(layer(x))
#             x = self.dropout(x)
#         x = self.output_layer(x)  # No Sigmoid here if using BCEWithLogitsLoss
#         return x