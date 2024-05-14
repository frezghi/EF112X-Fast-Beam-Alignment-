import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d, complex_normalize

data_set_input = np.load('data/data_input_complex.npy')
data_set_label = np.load('data/data_label_complex.npy')
data_set_label_transmission = np.load('data/data_label_transmission_complex.npy')

# Calculate the mean absolute value (MAV)
mav = np.mean(np.abs(data_set_input))

# Divide each element of the data by the MAV
normalized_input = data_set_input / mav

# Determine the number of classes
num_classes = 64  # Assuming labels are 0-indexed

print(f'Number of classes will be: {num_classes}')

# Perform one-hot encoding
one_hot_labels = np.eye(num_classes)[data_set_label]

X_train, X_test, y_train, y_test, transmission_train, transmission_test = train_test_split(normalized_input, one_hot_labels, data_set_label_transmission, test_size=0.2, random_state=42)

hidden_size = 128

class ComplexNet(nn.Module):
    
    def __init__(self, input_size, num_classes, hidden_size):
        super(ComplexNet, self).__init__()
        self.fc1 = ComplexLinear(input_size, hidden_size)
        self.fc2 = ComplexLinear(hidden_size, hidden_size)
        self.fc3 = ComplexLinear(hidden_size, num_classes)
             
    def forward(self, x):
        x = self.fc1(x)
        x = complex_relu(x)
        x = self.fc2(x)
        x = complex_relu(x)
        x = self.fc3(x)
        x = x.abs()
        x = nn.functional.softmax(x, dim=1)
        return x

class ComplexDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transmissions):
        self.data = torch.tensor(data, dtype=torch.cfloat, requires_grad=True)  # Convert data to tensor of dtype=torch.cfloat
        self.labels = torch.tensor(labels, requires_grad=True)
        self.transmissions = torch.tensor(transmissions, requires_grad=True)
        #self.transmissions = transmissions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        transmission = self.transmissions[idx]
        return sample, label, transmission
    
def custom_loss(outputs, labels, transmissions):
        dim = 1
        power_product = torch.sum(transmissions*outputs, dim=dim)/torch.sum(transmissions*labels, dim=dim)
        #print(power_product)
        running_loss = torch.mean(power_product)
        #print(running_loss)
        mean_loss = 1-(running_loss)
        #print(mean_loss)
        return torch.tensor(mean_loss, requires_grad=True)


    
def test_power():
    model.eval()
    percent_diff_sum = 0.0
    num_samples = 0

    with torch.no_grad():
        for inputs, labels, transmission in zip(test_loader.dataset.data, test_loader.dataset.labels, test_loader.dataset.transmissions):
            inputs = inputs.view(1, -1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            predicted_channel = predicted.item()
            best_channel = np.argmax(labels).item()

            #print(inputs)
            channel_transmission = transmission[best_channel]

            try:
                final_prediction = max(transmission[predicted_channel-1], transmission[predicted_channel], transmission[predicted_channel+1])
            except IndexError:
                try:
                    final_prediction = max(transmission[predicted_channel-1], transmission[predicted_channel], transmission[predicted_channel-2])
                except IndexError:
                    final_prediction = max(transmission[predicted_channel+1], transmission[predicted_channel], transmission[predicted_channel+2])

            percent_diff = (final_prediction / transmission[best_channel]) * 100
            percent_diff_sum += percent_diff
            num_samples += 1

            #print(f"Predicted Channel: {predicted_channel}, Best Channel: {best_channel}")
            #print(f"Transmission Power from Predicted Channel: {final_prediction}, Max Transmission Power: {transmission[best_channel]}")
            #print(f"Percentage of total power: {percent_diff}%\n")

    average_percent_diff = percent_diff_sum / num_samples
    print(f'Average percent of power from predicted channel compared to best channel: {average_percent_diff}%')


# Initialize dataset and dataloader
train_dataset = ComplexDataset(X_train, y_train, transmission_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = ComplexDataset(X_test, y_test, transmission_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize the model
input_size = data_set_input.shape[1]
model = ComplexNet(input_size, num_classes, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda:0" )

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels, transmissions in train_loader:
        inputs = inputs.view(inputs.size(0), -1)  # Flatten each batch of inputs
        #inputs = inputs.unsqueeze(1)  # Add a channel dimension
        #inputs = inputs.unsqueeze(1)
        #inputs, labels = inputs.to(device), labels.to(device) # Train on gpu
        optimizer.zero_grad()
        outputs = model(inputs)
        #loss = criterion(outputs, labels)
        loss = custom_loss(outputs, labels, transmissions)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

