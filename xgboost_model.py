import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Generate_dataset import generate_data
from sklearn.preprocessing import LabelEncoder


num_antenna_bs = 64
data_set_input, data_set_label = generate_data(num_samples=1000, num_antenna=num_antenna_bs)

# Reshape input data for XGBoost
X = data_set_input.reshape(len(data_set_input), -1)
print(X[0:2].shape)

#label_encoder = LabelEncoder()
#label_encoder.classes_ = np.arange(num_antenna_bs)
#y = label_encoder.fit_transform(data_set_label)

y = []

for label in data_set_label:
    row = np.arange(num_antenna_bs)
    row[label] = 1
    y.append(row)

y = np.array(y)
print(y.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost model
model = XGBClassifier(objective='multi:softmax', num_class=num_antenna_bs, eval_metric='mlogloss')

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Save the model
#pickle.dump(model, open("xgboost_model.pkl", "wb"))
