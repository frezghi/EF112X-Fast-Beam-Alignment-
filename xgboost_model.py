import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from Generate_dataset import generate_data
from sklearn.preprocessing import LabelEncoder


num_antenna_bs = 32
#data_set_input, data_set_label = generate_data(num_samples=100000, num_antenna=num_antenna_bs)

data_set_input = np.load('data/data_input.npy')
data_set_label = np.load('data/data_label.npy')
data_set_label_transmission = np.load('data/data_label_transmission.npy')

# Reshape input data for XGBoost
X = data_set_input.reshape(len(data_set_input), -1)


# Adds dummy data for missing classes since XGB requires each class to appear atleast once in training set
# Adds the data twice since it will be split into train/test with stratify enabled
for i in range(num_antenna_bs):
    if i not in data_set_label:
        data_set_label = np.append(data_set_label, [i,i])
        similar_index = np.argwhere(data_set_label == i-1)[0]
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


#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Save the model
#pickle.dump(model, open("xgboost_model.pkl", "wb"))
