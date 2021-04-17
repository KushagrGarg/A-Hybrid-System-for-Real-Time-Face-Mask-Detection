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
from keras.models import load_model
import seaborn as sns
from sklearn.metrics import f1_score,confusion_matrix

class model2:
    def __init__(self,pathdata,pathlabel):
        infile = open(pathdata,'rb')
        self.X = pkl.load(infile)
        infile.close()

        infile = open(pathlabel,'rb')
        self.y = pkl.load(infile)
        infile.close()
        print(self.X.shape,self.y.shape)
    def CNN(self):
       m= load_model('model-012.model/')
       new_model = Sequential()
       for layer in m.layers[:-1]: # excluding last layer from copying
           new_model.add(layer)
       self.mod=Sequential()
       for layer in new_model.layers[:-1]: # excluding last layer from copying
           self.mod.add(layer)
    def Summary(self):
        print(self.mod.summary())

    def SVM_PCATrain(self):
        print("*****Training Using SVM**************")
        features=self.mod.predict(self.X)
        pca = PCA(n_components=256)
        pca.fit(features)
        PCA_features = pca.transform(features)
        print(PCA_features.shape)
        self.X_train_svm,self.X_test_svm, self.y_train_svm, self.y_test_svm = train_test_split(PCA_features,self.y, test_size=0.2, random_state=42)
        self.classifier = svm.SVC(kernel='rbf', probability=True)
        self.classifier.fit(self.X_train_svm,self.y_train_svm)

    def TestAccuracy(self):
        predy=self.classifier.predict(self.X_test_svm)
        print("Test Accuracy Using SVM:",np.sum(1*(predy==self.y_test_svm))/self.y_test_svm.shape[0])
        cm=confusion_matrix(self.y_test_svm,predy)
