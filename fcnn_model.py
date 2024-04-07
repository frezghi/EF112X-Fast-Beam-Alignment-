import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

time_slots = 4
num_antenna = 32

data_set_input = np.load('data/data_input_complex.npy')
data_set_label = np.load('data/data_label_complex.npy')
data_set_label_transmission = np.load('data/data_label_transmission_complex.npy')

X_train, X_test, y_train, y_test, transmission_train, transmission_test = train_test_split(data_set_input, data_set_label, data_set_label_transmission, test_size=0.2, random_state=42)


class ComplexFCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ComplexFCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ComplexDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.cfloat)  # Convert data to tensor of dtype=torch.cfloat
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


# Initialize dataset and dataloader
train_dataset = ComplexDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = ComplexDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model
input_size = time_slots
hidden_size = 128 
num_classes = num_antenna
model = ComplexFCNN(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.view(inputs.size(0), -1)  # Flatten each batch of inputs
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# Evaluation
model.eval()
percent_diff_sum = 0.0
num_samples = 0

with torch.no_grad():
    for inputs, labels, transmission in zip(test_loader.dataset.data, test_loader.dataset.labels, transmission_test):
        inputs = inputs.view(1, -1)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        predicted_channel = predicted.item()
        best_channel = labels.item()
        channel_transmission = transmission[best_channel]

        percent_diff = (channel_transmission[predicted_channel] / max(channel_transmission)) * 100
        percent_diff_sum += percent_diff
        num_samples += 1

        print(f"Predicted Channel: {predicted_channel}, Best Channel: {best_channel}")
        print(f"Transmission Power from Predicted Channel: {channel_transmission[predicted_channel]}, Max Transmission Power: {max(channel_transmission)}")
        print(f"Percentage Difference: {percent_diff}%\n")

average_percent_diff = percent_diff_sum / num_samples
print(f'Average percent of power from predicted channel compared to best channel: {average_percent_diff}%')

