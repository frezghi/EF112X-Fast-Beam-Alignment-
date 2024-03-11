"""
Author: Yuqiang (Ethan) Heng
"""
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Model
from keras.layers import Input
from keras import backend as K
from keras.utils import np_utils

def cart2sph(xyz,center):
    x = np.subtract(xyz[:,0],center[0])
    y = np.subtract(xyz[:,1],center[1])
    z = np.subtract(xyz[:,2],center[2])
    rtp = np.zeros(xyz.shape)
    r = np.sqrt(np.power(x,2)+np.power(y,2)+np.power(z,2))
    theta = np.arccos(np.divide(z,r))
    phi = np.arctan2(y,x)
    rtp[:,0] = r
    rtp[:,1] = theta
    rtp[:,2] = phi
    return rtp

#dataset = np.load('./Dataset/MISO_AP_selection_processed_cartesian.npy')
#X_cart = dataset[:,:3]
X_cart = 0
X_cart = X_cart + np.random.normal(0,2.0409,X_cart.shape)
nap = 16
ap_coordinates = np.zeros((nap,3))
for i in range(4):
    for j in range(4):
        ap_coordinates[i*4+j,0] = 506.743561 + 75*j
        ap_coordinates[i*4+j,1] = 426.086060 + 75*i
X = np.concatenate([cart2sph(X_cart,ap_coordinates[i,:]) for i in range(nap)],axis=1)
        
#Y = dataset[:,3]
Y = 0
# encode class values as integers and convert to one-hot vectors
encoder = LabelEncoder()
encoder.fit(Y)
Y_onehot = np_utils.to_categorical(encoder.transform(Y))
# 0.8/0.1/0.1 train/val/test split
X_train, X_tmp, Y_train, Y_tmp = train_test_split(X, Y_onehot, test_size=0.2)
X_val, X_test, Y_val, Y_test = train_test_split(X_tmp, Y_tmp, test_size=0.5)

K.clear_session()
input_location = Input(shape=(X_train.shape[1],))
dense1 = Dense(48, activation ='sigmoid')(input_location)
dense2 = Dense(48, activation = 'sigmoid')(dense1)
dense3 = Dense(32, activation = 'sigmoid')(dense2)
dense4 = Dense(24, activation = 'sigmoid')(dense3)
output = Dense(nap, activation = 'softmax')(dense4)
model = Model(inputs = input_location, outputs = output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data = (X_val, Y_val), epochs = 20, batch_size = 100)