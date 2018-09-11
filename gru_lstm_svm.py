
# coding: utf-8

# ### Build RNN/LSTM model

# In[41]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import keras
import os
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,LSTM, Activation, GRU,SimpleRNN, BatchNormalization, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC, SVC
#print(keras.__version__)


# ## Data preprocessing

# In[2]:

# Load datafile name and path 
data_paths = []
for root, dirs, files in os.walk('./biosignals_filtered'):
    for name in files:
        if name[-3:] == 'csv':
            paths = os.path.join(root, name)
            data_paths.append(paths)
# len(data_paths): 8600
    


# In[3]:

# checked all data point shape is (2816,6) and without missing values
# for csv in data_paths:
#     data = pd.read_csv(datacsv, sep = '\t') 
#     if data.shape != (2816,6) or data.isnull().sum().sum() >0 :
#         print(csv)
    


# In[4]:

# extract target y label from the csv filename
# csv = './biosignals_filtered/071309_w_21/071309_w_21-BL1-081_bio.csv'
def get_target(csv, text = False):
    """
    Extract target y label from the csv filename string
    Input: 
        csv: file name string.
        text: whether return text 'PA1-4"/"BL1" or number 0-4
    Output: target y label number maps 'BL1', 'PA1', 'PA2', 'PA3','PA4']
    """
    y_mapping = {'BL1':0, 'PA1':1, 'PA2':2, 'PA3':3,'PA4':4}
    
    idx = max( csv.find('PA'), csv.find('BL'))
    label = csv[idx:idx+3]
    if text:
        return label
    return y_mapping[label]

# LABELS = ['BL1', 'PA1', 'PA2', 'PA3','PA4']
# target = []
# # checked extract labels correctly
# for i in data_paths:
#     tar = get_target(i)
#     target.append(tar)
#     if tar not in LABELS:
#         print(i, tar)
# import collections
# counter=collections.Counter(target)
# counter
# Counter({'BL1': 1720, 'PA1': 1720, 'PA2': 1720, 'PA3': 1720, 'PA4': 1720})


# In[5]:

# seperate target=[0-4] groups
target0_path, target1_path, target2_path, target3_path, target4_path = [], [], [], [], []
add_dict = {0: (lambda x: target0_path.append(x)), 
           1: (lambda x: target1_path.append(x)), 
           2: (lambda x: target2_path.append(x)), 
           3: (lambda x: target3_path.append(x)), 
           4: (lambda x: target4_path.append(x))}
for path in data_paths:
    target = get_target(path)
    add_dict[target](path)
    
#print(len(target0_path), len(target1_path),len(target2_path),len(target3_path),len(target3_path))


# In[6]:

# make input dataset
def generate_input_list(groups):
    """
    Generate input dataset path list, give tuple of target label.
    Input: groups, tuple of xx vs. xx for binary/multi classfication, i.e.: (0,1) or (0,1,4)
    Output: csv file name and path list
    """
    lst = []
    name_map = {0: target0_path, 1: target1_path, 2:target2_path, 3:target3_path, 4:target4_path}
    for i in groups:
        lst += name_map[i]
    return lst


# ### load dataset into memory, nomalization
# 

# In[32]:

# define the classification: binary or multiclassfication, which class label to recognize


def generate_data(groups):
    """
    Generate 3D input for LSTM. X value is normalizaed.
    Input: 
        groups, tuple of xx vs. xx for binary/multi classfication, 
        i.e.: (0,1) or (0,1,4)
    Output: 
        X: ndarray of shape (samples, time steps, and features)
        y: ndarray of shape (samples, 1)
    """
    # get path list for the intended classification problem
    input_paths = generate_input_list(groups) 
    X_lst = []
    y = []
    for p in input_paths:
        dp = pd.read_csv(p, sep = '\t') #datapoint
        # Normalization 
        # norm = lambda x: (x - x.mean()) / x.std()
        # dp = dp.apply(norm)
        # Min-Max scaling 
        #dp_norm = (dp - dp.min()) / (dp.max() - dp.min())
        #dp = dp_norm.values
        if dp.isnull().sum().sum()>0:
#             print(p, dp.isnull().sum().sum())
            continue
        dp = dp.drop(['time'], axis = 1)     
        dp = dp.iloc[:1600:4]

        if dp.isnull().sum().sum()>0:
#             print('after norm',p, dp.isnull().sum().sum())
            continue
        dp = dp.values

        X_lst.append(dp)
        sample_y = get_target(p, text= True)
        y.append(sample_y)
    X = np.stack(X_lst, axis=0)
    
    # convert y into int 0 and 1
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    y_dummy = y
    # convert y into one-hot encoding
    if len(groups)>2:
        y_dummy = pd.get_dummies(y)
        y_dummy = y_dummy.values
    return X, y , y_dummy
    


# In[30]:

#generate_data((0,1,4))


# In[8]:

# X, y = generate_data((0,4)) # for BL1 vs PA4 binary classification
# print("X shape: ", X.shape, '\ny shape: ', y.shape)


# In[9]:

# split into train and validation datasets
def tr_val_data(groups):
    X, y, y_dummy = generate_data(groups) # for BL1 vs PA4 binary classification
    print(' ')
    print('=============>data group', groups)
    #print("X shape: ", X.shape, '\ny shape: ', y.shape)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y_dummy, test_size=0.1, shuffle = True)
    #print("X_train shape: ", X_tr.shape, "y_train shape: ", y_tr.shape)
    #print('X_test shape:', X_val.shape, '  y_test shape:', y_val.shape)
    #print(' ')
    return X, y, y_dummy, X_tr, X_val, y_tr, y_val




# ## Build LSTM model

# In[10]:

def rnn(num_groups):
    model = Sequential()
    model.add(GRU(128,
                   input_shape = (T, D),
                   return_sequences=True,
                   dropout=0.2, 
                   recurrent_dropout=0.2))
    model.add(LSTM(128,
                   return_sequences=False,
                   dropout=0.2, 
                   recurrent_dropout=0.2))
    # model.add(LSTM(8,
    #                return_sequences=False,
    #                dropout=0.2, 
    #                recurrent_dropout=0.2))
    if num_groups == 2:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.add(Dense(num_groups, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[46]:

T, D = 400, 5

def train(groups):
    X, y, y_dummy, X_tr, X_val, y_tr, y_val = tr_val_data(groups)
    model = rnn(len(groups))
    model.fit(X_tr, y_tr, batch_size = 16, epochs = 5, 
          validation_data=(X_val, y_val), shuffle = True, verbose =1)
    score, acc = model.evaluate(X_val, y_val)
    print ("")
    print ('Test accuracy for group',groups, ' is: ', acc)
    return model, X, y, y_dummy


# ### Main running func

# In[37]:

# model1 = train((0,1))
# model2 = train((0,2))
# model3 = train((0,3))
# model4 = train((0,4))
# model014 = train((0,1,4))
# model_all = train((0,1,2,3,4))


# ### Combine the last layer with SVM
# 
# Ref:
# 
# https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer
# 
# 
# https://stackoverflow.com/questions/40401008/keras-neural-networks-and-sklearn-svm-svc
# 
# About implementation, you just have to train a neural network, then select one of the layers (usually the ones right before the fully connected layers or the first fully connected one), run the neural network on your dataset, store all the feature vectors, then train an SVM with a different library (e.g sklearn).
# 

# In[47]:

def svm_layer(groups):
    model, X, y, y_dummy= train(groups)
    get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                      [model.layers[1].output])
    
    # output in test mode = 0
    layer_output = get_3rd_layer_output([X, 0])[0]
    
    # output in train mode = 1
    #layer_output_train = get_3rd_layer_output([X, 1])[0]
    
    print("Shape of svm layer: ", layer_output.shape)
    
    X_svm = layer_output
    svm_estimator = SVC()
    svm_result = cross_validate(svm_estimator, X_svm, y, cv=10, return_train_score=True)
    train_accu = svm_result['train_score'].mean()
    test_accu = svm_result['test_score'].mean()
    print('SVM Train accuracy: %.2f%%, Test accuracy: %.2f%%' %( train_accu*100,test_accu*100))


# In[ ]:

# svm_layer((0,1))
# svm_layer((0,2))
# svm_layer((0,3))
# svm_layer((0,4))
# svm_layer((0,1,2,3,4))
svm_layer((0,1,4))