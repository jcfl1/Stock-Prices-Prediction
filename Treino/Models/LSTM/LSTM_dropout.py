import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the LSTM model
class LSTMModel_dropout(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device= 'cpu'):
        super(LSTMModel_dropout, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p= .3)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation to convert logits to probabilities
        self.device = device
        self.to(device)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze(1) # output has only batch dimension

    def fit(self, X_train, y_train, learning_rate, num_epochs = 50):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype= torch.float32).to(self.device),
                                                torch.tensor(y_train, dtype= torch.float32).to(self.device)),
                                                batch_size=32,
                                                shuffle=True)

        # Training loop
        self.train()
        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
    
    def predict(self, X, threshold=0.5):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            # Convert the input numpy array to a PyTorch tensor
            X_tensor = torch.tensor(X, dtype=torch.float32)
            # Move the tensor to the appropriate device (CPU or GPU)
            device = next(self.parameters()).device
            X_tensor = X_tensor.to(device)
            # Perform a forward pass through the model
            logits = self.forward(X_tensor)
            probabilities = self.sigmoid(logits)  # Apply sigmoid activation
            predictions = (probabilities > threshold).float()  # Thresholding to get 0 or 1
        # Convert predictions to numpy array and return
        return predictions.cpu().numpy().astype(int)


