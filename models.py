import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.decomposition import PCA
from sklearn import svm
from matplotlib import pyplot as plt
import pickle as pkl
from sklearn.model_selection import train_test_split
def plot(history):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(history.history['loss'],'r',label='training loss')
    ax1.plot(history.history['val_loss'],label='validation loss')
    ax1.set_xlabel('# epochs')
    ax1.set_ylabel('loss')
    ax1.legend()
    ax2.plot(history.history['accuracy'],'r',label='training accuracy')
    ax2.plot(history.history['val_accuracy'],label='validation accuracy')
    ax2.set_xlabel('# epochs')
    ax2.set_ylabel('loss')
    ax2.legend()
class Model1:
    def __init__(self,pathdata,pathlabel):
        infile = open(pathdata,'rb')
        self.X = pkl.load(infile)
        infile.close()

        infile = open(pathlabel,'rb')
        self.y = pkl.load(infile)
        infile.close()
        print(self.X.shape,self.y.shape)
    def CNNModel(self):
        #X=np.load('data.npy')
        #y=np.load('target.npy')
        self.model=Sequential()
        self.model.add(Conv2D(200,(3,3),input_shape=self.X.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(100,(3,3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(50,activation='relu'))
        self.model.add(Dense(2,activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        print(self.model.summary())
    
    def trainCNNNmodel(self):
        self.train_X,self.test_X,self.train_y,self.test_y=train_test_split(self.X,self.y,test_size=0.1)
        checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
        self.history=self.model.fit(self.train_X,self.train_y,epochs=20,callbacks=[checkpoint],validation_split=0.2)
        return self.history
    def plot(self):
        plot(self.history)

    def evalutae(self):
        print(self.model.evaluate(self.test_X,self.test_y))
    
