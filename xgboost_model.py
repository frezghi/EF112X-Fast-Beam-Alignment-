import numpy as np
from xgboost import XGBClassifier, QuantileDMatrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from Generate_dataset import generate_data
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import optuna
import xgboost as xgb

# Trial 75 finished with value: 0.8798876534547864 and parameters: 
# {'top_k': 4, 'learning_rate': 0.002570851186896932, 'max_depth': 6, 'min_child_weight': 3.643498758169179, 'subsample': 0.7248408437172833, 'colsample_bytree': 0.8231718582302777, 'lambda': 0.05712353470876617, 'alpha': 0.00322329095620129, 'n_estimators': 361}

# Trial 788 finished with value: 0.8864763463737456 and parameters: 
# {'top_k': 4, 'learning_rate': 0.023112609436983768, 'max_depth': 6, 'min_child_weight': 3.391841137450273, 'subsample': 0.7012262317751768, 'colsample_bytree': 0.9097544503574357, 'lambda': 0.0031252177968801517, 'alpha': 0.9361220214755489, 'n_estimators': 342}

# Trial 868 finished with value: 0.8897817995408716 and parameters: 
# {'learning_rate': 0.025761751766574758, 'max_depth': 10, 'min_child_weight': 3.0167571326587956, 'subsample': 0.5464807939759584, 'colsample_bytree': 0.8819409207447095, 'lambda': 0.007491224544185017, 'alpha': 0.543762825977817, 'n_estimators': 322}

# Trial 130 finished with value: 0.8959910834279285 and parameters: 
# {'learning_rate': 0.11487685387942768, 'reg_alpha': 2.6823649613473153, 'reg_lambda': 2.2455247021025597, 'n_estimators': 375, 'min_child_weight': 9, 'max_depth': 7, 'gamma': 0.24937282763295612, 'colsample_bytree': 0.5708261180238751, 'subsample': 0.973475858836052}

def main():
    train(num_antenna_bs=100, time_slots=6, top_k=4)
    return

    num_antenna_bs = 100
    time_slots = 10
    top_k = 4

    data_set_input = np.load('data/data_input.npy')
    data_set_label = np.load('data/data_label.npy')
    data_set_label_transmission = np.load('data/data_label_transmission.npy')

    Qmatrix_train, X_test, y_test, transmission_test = preprocess_data(data_set_input, data_set_label, data_set_label_transmission, num_antenna_bs)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, Qmatrix_train, X_test, y_test, transmission_test, num_antenna_bs, time_slots, top_k, False), n_trials=500)


    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    #train(num_antenna_bs=200, time_slots=17, top_k=3)
    return
    for k in [30, 60, 100, 140, 180, 240, 320, 410, 520]:
        train(num_antenna_bs=k, time_slots=int((k/10)), top_k=0)

def preprocess_data(data_set_input, data_set_label, data_set_label_transmission, num_antenna_bs):
    X = data_set_input.reshape(len(data_set_input), -1)

    # data manipulation
    for i in range(num_antenna_bs):
            if np.count_nonzero(data_set_label == i)<2:
                data_set_label = np.append(data_set_label, [i, i])
                try:
                    similar_indices = np.argwhere(data_set_label == i - 1)
                    similar_indices = np.append(similar_indices, np.argwhere(np.isin(data_set_label, [i-2, i-3, i-4, i-5, i-6, i-7, i-8, i-9, i-10])))
                    similar_index = similar_indices[0]
                    #print(data_set_label[similar_index], i)
                    X = np.append(X, X[similar_index], axis=0)
                    X = np.append(X, X[similar_index], axis=0)
                    data_set_label_transmission = np.append(data_set_label_transmission, data_set_label_transmission[similar_index], axis=0)
                    data_set_label_transmission = np.append(data_set_label_transmission, data_set_label_transmission[similar_index], axis=0)
                except:
                    similar_indices = np.argwhere(data_set_label == i + 1)
                    similar_indices = np.append(similar_indices, np.argwhere(np.isin(data_set_label, [i+2, i+3, i+4, i+5, i+6, i+7, i+8, i+9, i+10])))
                    similar_index = 0
                    X = np.append(X, np.expand_dims(X[similar_index], axis=0), axis=0)
                    X = np.append(X, np.expand_dims(X[similar_index], axis=0), axis=0)
                    data_set_label_transmission = np.append(data_set_label_transmission, np.expand_dims(data_set_label_transmission[similar_index], axis=0), axis=0)
                    data_set_label_transmission = np.append(data_set_label_transmission, np.expand_dims(data_set_label_transmission[similar_index], axis=0), axis=0)

    X_train, X_test, y_train, y_test, transmission_train, transmission_test = train_test_split(X, data_set_label, data_set_label_transmission, test_size=0.05, random_state=42, stratify=data_set_label)
    Qmatrix_train = QuantileDMatrix(X_train, y_train)
    
    return Qmatrix_train, QuantileDMatrix(X_test), y_test, transmission_test

def objective(trial, Qmatrix_train, X_test, y_test, transmission_test, num_antenna_bs, time_slots, top_k, first_stage):    
    if first_stage == True:
        top_k = trial.suggest_int('top_k', 1, time_slots-4)

    # learning_rate = trial.suggest_float("learning_rate", 1e-3, 1, log=True)
    # max_depth = trial.suggest_int("max_depth", 3, 10)
    # min_child_weight = trial.suggest_float("min_child_weight", 0.1, 10)
    # subsample = trial.suggest_float("subsample", 0.5, 1.0)
    # colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    # reg_lambda = trial.suggest_float("lambda", 1e-3, 10.0, log=True)
    # reg_alpha = trial.suggest_float("alpha", 1e-3, 10.0, log=True)
    # n_estimators = trial.suggest_int("n_estimators", 50, 500)

    param = {
        'objective': 'multi:softprob',
        'verbosity': 0,
        'tree_method': 'hist',
        'device': 'cuda',
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 10),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'gamma': trial.suggest_float('gamma', 0.001, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1),
        'subsample' : trial.suggest_float("subsample", 0.5, 1.0),
        'num_class': num_antenna_bs
    }
    n_estimators = param.pop('n_estimators')

    if first_stage == True:
        data_set_input, data_set_label, data_set_label_transmission = generate_data(num_samples=5000, num_antenna_bs=num_antenna_bs, time_slots=(time_slots-top_k))
        X = data_set_input.reshape(len(data_set_input), -1)

        # data manipulation
        for i in range(num_antenna_bs):
                if np.count_nonzero(data_set_label == i)<2:
                    data_set_label = np.append(data_set_label, [i, i])
                    try:
                        similar_indices = np.argwhere(data_set_label == i - 1)
                        similar_indices = np.append(similar_indices, np.argwhere(np.isin(data_set_label, [i-2, i-3, i-4, i-5, i-6, i-7, i-8, i-9, i-10])))
                        similar_index = similar_indices[0]
                        #print(data_set_label[similar_index], i)
                        X = np.append(X, X[similar_index], axis=0)
                        X = np.append(X, X[similar_index], axis=0)
                        data_set_label_transmission = np.append(data_set_label_transmission, data_set_label_transmission[similar_index], axis=0)
                        data_set_label_transmission = np.append(data_set_label_transmission, data_set_label_transmission[similar_index], axis=0)
                    except:
                        similar_indices = np.argwhere(data_set_label == i + 1)
                        similar_indices = np.append(similar_indices, np.argwhere(np.isin(data_set_label, [i+2, i+3, i+4, i+5, i+6, i+7, i+8, i+9, i+10])))
                        similar_index = 0
                        X = np.append(X, np.expand_dims(X[similar_index], axis=0), axis=0)
                        X = np.append(X, np.expand_dims(X[similar_index], axis=0), axis=0)
                        data_set_label_transmission = np.append(data_set_label_transmission, np.expand_dims(data_set_label_transmission[similar_index], axis=0), axis=0)
                        data_set_label_transmission = np.append(data_set_label_transmission, np.expand_dims(data_set_label_transmission[similar_index], axis=0), axis=0)

        X_train, X_test, y_train, y_test, transmission_train, transmission_test = train_test_split(X, data_set_label, data_set_label_transmission, test_size=0.2, random_state=42, stratify=data_set_label)

    model = xgb.train(param, Qmatrix_train, num_boost_round = n_estimators)

    y_probs = model.predict(X_test)

    curr_sum_topk = 0
    num_elements = 0 

    for probs, best_channel, channel_transmission in zip(y_probs, y_test, transmission_test):
        sorted_indices = np.argsort(probs)[::-1]
        top_k_indices = sorted_indices[:top_k+1]
        max_k_prediction = max(channel_transmission[top_k_indices])

        curr_sum_topk += max_k_prediction / channel_transmission[best_channel]
        num_elements += 1

    return curr_sum_topk / num_elements

def train(num_antenna_bs, time_slots, top_k):
    data_set_input, data_set_label, data_set_label_transmission = generate_data(num_samples=500000, num_antenna_bs=num_antenna_bs, time_slots=time_slots)

    #data_set_input = np.load('data/data_input.npy')
    #data_set_label = np.load('data/data_label.npy')
    #data_set_label_transmission = np.load('data/data_label_transmission.npy')

    # Reshape input data for XGBoost
    X = data_set_input.reshape(len(data_set_input), -1)

    # Adding samples for labels with less than 2 samples (required with XGBoost)
    for i in range(num_antenna_bs):
            if np.count_nonzero(data_set_label == i)<2:
                data_set_label = np.append(data_set_label, [i, i])
                try:
                    similar_indices = np.argwhere(data_set_label == i - 1)
                    similar_indices = np.append(similar_indices, np.argwhere(np.isin(data_set_label, [i-2, i-3, i-4, i-5, i-6, i-7, i-8, i-9, i-10])))
                    similar_index = similar_indices[0]
                    #print(data_set_label[similar_index], i)
                    X = np.append(X, X[similar_index], axis=0)
                    X = np.append(X, X[similar_index], axis=0)
                    data_set_label_transmission = np.append(data_set_label_transmission, data_set_label_transmission[similar_index], axis=0)
                    data_set_label_transmission = np.append(data_set_label_transmission, data_set_label_transmission[similar_index], axis=0)
                except:
                    similar_indices = np.argwhere(data_set_label == i + 1)
                    similar_indices = np.append(similar_indices, np.argwhere(np.isin(data_set_label, [i+2, i+3, i+4, i+5, i+6, i+7, i+8, i+9, i+10])))
                    similar_index = 0
                    X = np.append(X, np.expand_dims(X[similar_index], axis=0), axis=0)
                    X = np.append(X, np.expand_dims(X[similar_index], axis=0), axis=0)
                    data_set_label_transmission = np.append(data_set_label_transmission, np.expand_dims(data_set_label_transmission[similar_index], axis=0), axis=0)
                    data_set_label_transmission = np.append(data_set_label_transmission, np.expand_dims(data_set_label_transmission[similar_index], axis=0), axis=0)

    y = data_set_label
    X_train, X_test, y_train, y_test, transmission_train, transmission_test = train_test_split(X, y, data_set_label_transmission, test_size=0.2, random_state=42, stratify=y)

    # Define the XGBoost model
    params = {'learning_rate': 0.11487685387942768, 'reg_alpha': 2.6823649613473153, 'reg_lambda': 2.2455247021025597, 'n_estimators': 375, 'min_child_weight': 9, 'max_depth': 7, 'gamma': 0.24937282763295612, 'colsample_bytree': 0.5708261180238751, 'subsample': 0.973475858836052}
    model = XGBClassifier(objective='multi:softprob', **params)
    model.fit(X_train, y_train)

    #Make predictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)

    # Testing Phase
    # Evaluating percentage of transmission rate perserved
    curr_sum_neighbor = 0
    curr_sum_topk = 0
    num_elements = 0 
    for predicited_channel, probs, best_channel, channel_transmission in zip(y_pred, y_probs, y_test, transmission_test):
        # Get the indices that would sort y_pred in descending order
        sorted_indices = np.argsort(probs)[::-1]
        # Pick the top k indices with the highest probabilities
        top_k_indices = sorted_indices[:top_k+1]
        max_k_prediction = 0
        for k_index in top_k_indices:
            max_k_prediction = max(max_k_prediction, channel_transmission[k_index])

        # Search neighbouring channels, but make sure to stay in bounds of array
        try:
            neighbor_prediction = max(channel_transmission[predicited_channel-1], channel_transmission[predicited_channel], channel_transmission[predicited_channel+1])
        except IndexError:
            try:
                neighbor_prediction = max(channel_transmission[predicited_channel-1], channel_transmission[predicited_channel], channel_transmission[predicited_channel-2])
            except IndexError:
                neighbor_prediction = max(channel_transmission[predicited_channel+1], channel_transmission[predicited_channel], channel_transmission[predicited_channel+2])

        #print(channel_transmission)
        #print(neighbor_prediction, channel_transmission[best_channel])

        curr_sum_neighbor += neighbor_prediction/channel_transmission[best_channel]
        curr_sum_topk += max_k_prediction/channel_transmission[best_channel]
        num_elements += 1

    print(f'Average percent of power from predicted channel compared to best channel using neighbor {100*curr_sum_neighbor/num_elements}%')
    print(f'Average percent of power from predicted channel compared to best channel using top_k {100*curr_sum_topk/num_elements}%')

    curr_sum2 = 0
    num_elements2 = 0

    for predicited_channel, best_channel, channel_transmission in zip(y_pred, y_test, transmission_test):
        curr_sum2 += channel_transmission[predicited_channel]/max(channel_transmission)
        num_elements2 += 1


    print(f'Average percent of power from predicted channel compared to best channel using no extra signals: {100*curr_sum2/num_elements2}%')

main()
exit()

x = [0, 1, 2, 3, 4, 5, 6, 7, 8 ,9]
y = [0.69, 0.80, 0.86, 0.84, 0.88, 0.81, 0.80, 0.83, 0.80, 0.79]
plt.plot(x, y, marker='o', linestyle='-')
plt.xlabel('Number of probing signals assigned to top k search (out of 10)')
plt.ylabel('Power of predicted channel relative to optimal choice')
plt.title('Tradeoff between more input to model vs more top k searches\n    Number of antennas = 100')
plt.grid(True)
plt.xticks(range(len(x)), x)
plt.show()

x = [0, 1, 2, 3, 4, 5, 6, 7, 8 ,9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
y = [0.67, 0.74, 0.858, 0.874, 0.9, 0.92, 0.894, 0.9, 0.903, 0.878, 0.835, 0.869, 0.825, 0.865, 0.865, 0.816, 0.814, 0.839, 0.8, 0.71]
plt.plot(x, y, marker='o', linestyle='-')
plt.xlabel('Number of probing signals assigned to top k search (out of 20)')
plt.ylabel('Power of predicted channel relative to optimal choice')
plt.title('Tradeoff between more input to model vs more top k searches\n    Number of antennas = 200')
plt.grid(True)
plt.xticks(range(len(x)), x)
plt.show()

x = [30, 60, 100, 140, 180, 240, 320, 410, 520]
y = [0.767, 0.775, 0.842, 0.821, 0.872, 0.89, 0.902, 0.929, 0.93]
plt.plot(x, y, marker='o', linestyle='-')
plt.xlabel('Number of antenna')
plt.ylabel('Power of predicted channel relative to optimal choice')
plt.title('Model perfromance scaling with number of antenna using top k highest probability channels')
plt.grid(True)
#plt.xticks(range(len(x)), x)
plt.show()

x = [30, 60, 100, 140, 180, 240, 320, 410, 520]
y = [0.815, 0.756, 0.869, 0.862, 0.866, 0.77, 0.758, 0.762, 0.768]
plt.plot(x, y, marker='o', linestyle='-')
plt.xlabel('Number of antenna')
plt.ylabel('Power of predicted channel relative to optimal choice')
plt.title('Model perfromance scaling with number of antenna using neighbour search')
plt.grid(True)
#plt.xticks(range(len(x)), x)
plt.show()

x = [30, 60, 100, 140, 180, 240, 320, 410, 520]
y = [0.75, 0.69, 0.68, 0.672, 0.674, 0.668, 0.668, 0.663, 0.667]
plt.plot(x, y, marker='o', linestyle='-')
plt.xlabel('Number of antenna')
plt.ylabel('Power of predicted channel relative to optimal choice')
plt.title('Model perfromance scaling with number of antenna using no extra search heuristic')
plt.grid(True)
#plt.xticks(range(len(x)), x)
plt.show()

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