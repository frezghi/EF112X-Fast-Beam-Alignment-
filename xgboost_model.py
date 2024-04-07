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

# y = []

# for label in data_set_label:
#     row = np.arange(num_antenna_bs)
#     row[label] = 1
#     y.append(row)

# y = np.array(y)

le = LabelEncoder()
y = le.fit_transform(data_set_label)

X_train, X_test, y_train, y_test, transmission_train, transmission_test = train_test_split(X, y, data_set_label_transmission, test_size=0.2, random_state=42)

# Define the XGBoost model
print("Fit model")
model = XGBClassifier(objective='multi:softprob', num_class=num_antenna_bs)
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

curr_sum = 0
num_elements = 0 
for predicited_channel, best_channel, channel_transmission in zip(y_pred, y_test, transmission_test):
    print(channel_transmission[predicited_channel], max(channel_transmission))
    curr_sum += channel_transmission[predicited_channel]/max(channel_transmission)
    num_elements += 1

print(f'Average percent of power from predicted channel compared to best channel {100*curr_sum/num_elements}%')


#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Save the model
#pickle.dump(model, open("xgboost_model.pkl", "wb"))
