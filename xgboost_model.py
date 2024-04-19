import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from Generate_dataset import generate_data
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


num_antenna_bs = 32
#data_set_input, data_set_label = generate_data(num_samples=100000, num_antenna=num_antenna_bs)

data_set_input = np.load('data/data_input.npy')
data_set_label = np.load('data/data_label.npy')
data_set_label_transmission = np.load('data/data_label_transmission.npy')

print(np.shape(data_set_input), np.shape(data_set_label), np.shape(data_set_label_transmission))

# Reshape input data for XGBoost
X = data_set_input.reshape(len(data_set_input), -1)


# Adds dummy data for missing classes since XGB requires each class to appear atleast once in training set
# Adds the data twice since it will be split into train/test with stratify enabled
for i in range(num_antenna_bs):
    if i not in data_set_label:
        data_set_label = np.append(data_set_label, [i, i])
        similar_indices = np.argwhere(data_set_label == i - 1)
        if similar_indices.size > 0:
            similar_index = similar_indices[0]
            X = np.append(X, X[similar_index], axis=0)
            X = np.append(X, X[similar_index], axis=0)
            data_set_label_transmission = np.append(data_set_label_transmission, data_set_label_transmission[similar_index], axis=0)
            data_set_label_transmission = np.append(data_set_label_transmission, data_set_label_transmission[similar_index], axis=0)

#print(np.shape(X), np.shape(data_set_label), np.shape(data_set_label_transmission))

y = data_set_label

X_train, X_test, y_train, y_test, transmission_train, transmission_test = train_test_split(X, y, data_set_label_transmission, test_size=0.2, random_state=42, stratify=y)

# Define the XGBoost model
print("Fit model")
model = XGBClassifier(objective='multi:softprob', num_class=num_antenna_bs)
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

curr_sum = 0
num_elements = 0 
for predicited_channel, best_channel, channel_transmission in zip(y_pred, y_test, transmission_test):
    # Search neighbouring channels, but make sure to stay in bounds of array
    try:
        final_prediction = max(channel_transmission[predicited_channel-1], channel_transmission[predicited_channel], channel_transmission[predicited_channel+1])
    except IndexError:
        try:
            final_prediction = max(channel_transmission[predicited_channel-1], channel_transmission[predicited_channel], channel_transmission[predicited_channel-2])
        except IndexError:
            final_prediction = max(channel_transmission[predicited_channel+1], channel_transmission[predicited_channel], channel_transmission[predicited_channel+2])

    print(channel_transmission)
    print(final_prediction, channel_transmission[best_channel])

    curr_sum += final_prediction/channel_transmission[best_channel]
    num_elements += 1

print(f'Average percent of power from predicted channel compared to best channel {100*curr_sum/num_elements}%')

curr_sum2 = 0
num_elements2 = 0

for predicited_channel, best_channel, channel_transmission in zip(y_pred, y_test, transmission_test):
    curr_sum2 += channel_transmission[predicited_channel]/max(channel_transmission)
    num_elements2 += 1


print(f'Average percent of power from predicted channel compared to best channel using other method: {100*curr_sum2/num_elements2}%')


#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Save the model
#pickle.dump(model, open("xgboost_model.pkl", "wb"))

# Difference between best and predicted channel
"""
index_difference = np.abs(y_test - y_pred)

freq_difference = np.bincount(index_difference)

plt.figure(figsize=(10, 6))
plt.bar(range(len(freq_difference)), freq_difference, color='blue')
plt.xlabel('Difference in Index (Best - Predicted)')
plt.ylabel('Frequency')
plt.title('Frequency of Differences Between Best and Predicted Channels')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
"""
# Occurance of transmission speeds
"""
transmission_test_flat = transmission_test.flatten().astype(int)

transmission_speed_counts = np.bincount(transmission_test_flat)

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(transmission_speed_counts)), transmission_speed_counts[1:], color='blue')
plt.xlabel('Transmission Speed')
plt.ylabel('Occurrence')
plt.title('Occurrence of Transmission Speed')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(np.arange(1, len(transmission_speed_counts)))
plt.tight_layout()
plt.show()
"""
# Occurance of max transmission speed in every instance. Can be plotted for both 32 and 64 antennas.
"""
max_transmission_speeds = np.max(transmission_test, axis=1)
max_transmission_speed_counts = np.bincount(max_transmission_speeds.astype(int))

plt.figure(figsize=(10, 6))
plt.bar(range(len(max_transmission_speed_counts)), max_transmission_speed_counts, color='skyblue')
plt.xlabel('Maximum Transmission Speed')
plt.ylabel('Frequency')
plt.title('Frequency of Maximum Transmission Speeds')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(np.arange(len(max_transmission_speed_counts)))
plt.tight_layout()
plt.show()
"""