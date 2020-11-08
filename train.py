import pandas as pd
import numpy as np
import h5py
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import copy
import time
import random

#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#set_session(tf.Session(config=config))

with h5py.File(''.join(['eth.h5']), 'r') as hf:
    datas = hf['inputs'].value.astype(np.float32)
    labels = hf['outputs'].value.astype(np.float32)




#split training validation

print(labels.shape)

training_size = int(0.8* datas.shape[0])
training_datas = datas[:training_size,:]
training_labels = labels[:training_size,:]
validation_datas = datas[training_size:,:]
validation_labels = labels[training_size:,:]


class Trainer():
    def __init__(self):
        self.class_num = 9
        self.units = 50
        self.second_units = 30
        self.batch_size = 256
        self.epochs = 100
        self.output_size = 1
        self.reg = 1
        self.learning_rate= 0.01
        self.sample_prob = 0.2
        self.transaction_fee = 0.005
        self.prepare_data()
        self.init_network()

    def prepare_data(self):
        with h5py.File(''.join(['eth.h5']), 'r') as hf:
            datas = hf['inputs'].value.astype(np.float32)
            labels = hf['outputs'].value.astype(np.float32)
        new_labels = []
        for i,output in enumerate(labels):
            input = datas[i][-1][0]
            output = output[0]
            ratio = (output-input)/input
            if np.abs(ratio)<=self.transaction_fee:
                new_labels.append(0)
            elif ratio>0.03:
                new_labels.append(4)
            elif ratio>0.02:
                new_labels.append(3)
            elif ratio>0.01:
                new_labels.append(2)
            elif ratio>self.transaction_fee:
                new_labels.append(1)
            elif ratio<-0.03:
                new_labels.append(7)
            elif ratio<-0.02:
                new_labels.append(6)
            elif ratio<-0.01:
                new_labels.append(5)
            elif ratio<-self.transaction_fee:
                new_labels.append(4)
            else:
                new_labels.append(0)

        new_labels = np.array(new_labels)
        print(new_labels.shape)
        self.step_size = datas.shape[1]
        self.nb_features = datas.shape[2]

        datas = datas.reshape((datas.shape[0],-1))

        self.scalar = StandardScaler().fit(datas)
        datas = self.scalar.transform(datas)
        datas = datas.reshape((datas.shape[0],self.step_size,self.nb_features))

        training_size = int(0.8 * datas.shape[0])
        self.training_datas = datas[:training_size, :]
        self.training_labels = new_labels[:training_size]

        self.validation_datas = datas[training_size:, :]
        self.validation_labels = new_labels[training_size:]




        group = dict()
        for i, label in enumerate(self.training_labels):
            if group.get(label,None)==None:
                group[label] = list()
            group[label].append(self.training_datas[i])
        for key in group:
            print(key,len(group[key]))
        for key in group:
            if key!=0:
                group[key] =np.repeat(np.array(group[key]),int(len(group[0])//len(group[key])),axis=0)
        for key in group:
            print(key,len(group[key]))

        datas_new_labels=list()
        for key in group:
            for input in group[key]:
                datas_new_labels.append((input,key))
        random.shuffle(datas_new_labels)
        self.training_datas = [item[0] for item in datas_new_labels]
        self.training_labels = [item[1] for item in datas_new_labels]
        self.training_datas = np.array(self.training_datas )
        self.training_labels = np.array(self.training_labels)


        group = dict()
        for i, label in enumerate(self.validation_labels):
            if group.get(label,None)==None:
                group[label] = list()
            group[label].append(self.validation_datas[i])
        for key in group:
            print(key,len(group[key]))
        for key in group:
            if key!=0:
                group[key] =np.repeat(np.array(group[key]),int(len(group[0])//len(group[key])),axis=0)
        for key in group:
            print(key,len(group[key]))

        datas_new_labels=list()
        for key in group:
            for input in group[key]:
                datas_new_labels.append((input,key))
        random.shuffle(datas_new_labels)
        self.validation_datas = [item[0] for item in datas_new_labels]
        self.validation_labels = [item[1] for item in datas_new_labels]
        self.validation_datas = np.array(self.validation_datas )
        self.validation_labels = np.array(self.validation_labels)


        self.training_labels = np.eye(self.class_num)[self.training_labels]
        self.validation_labels = np.eye(self.class_num)[self.validation_labels]





    def init_network(self):

        def mean_squared_error(y_true, y_pred):
            y_pred = tf.convert_to_tensor(y_pred)
            y_true = tf.cast(y_true, y_pred.dtype)
            y_pred_index = tf.argmax(y_pred, axis=-1)
            y_true_index = tf.argmax(y_true, axis=-1)
            loss1 = tf.math.square(y_pred - y_true)
            loss2 = tf.cast(tf.math.square(y_pred_index - y_true_index),loss1.dtype)
            return tf.math.reduce_mean(loss1, axis=-1)+tf.math.reduce_mean(loss2,axis=-1)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(self.step_size,self.nb_features)))
        model.add(tf.compat.v1.keras.layers.LSTM(units=256,activity_regularizer=tf.keras.regularizers.l2(0.01),dropout=0.2))
        model.add(tf.keras.layers.Dense(self.class_num))
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Softmax())
        model.compile(optimizer="adam", loss="mse",metrics=['accuracy','mse'])
        self.model = model
        if os.path.exists("model/model"):
            self.model.load_weights("model/model")

    def train(self):
        self.model.fit(self.training_datas, self.training_labels, batch_size=self.batch_size,
                  validation_data=(self.validation_datas, self.validation_labels), epochs=self.epochs,
                  callbacks=[tf.keras.callbacks.ModelCheckpoint("model/model"),tf.keras.callbacks.Callback()])

    def predict(self):
        x = self.validation_datas[0:200,:]
        y = np.argmax(self.validation_labels[0:200,:],axis=-1)
        y_p = self.model.predict(
        x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10,
        workers=1, use_multiprocessing=False
        )
        y_p =  np.argmax(y_p,axis=-1)
        print(y)
        print(y_p)


train = Trainer()
train.train()

