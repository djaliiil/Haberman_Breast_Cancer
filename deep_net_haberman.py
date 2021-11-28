import tensorflow as tf
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from time import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LeakyReLU, ELU, PReLU
from tensorflow.keras.callbacks import EarlyStopping

#from keras.layers import Dense, Activation, LeakyReLU, ELU, PReLU
from tensorflow.keras.initializers import Constant
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam, SGD, Nadam
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
import category_encoders as ce
import pandas as pd
import numpy as np
from keras.utils.vis_utils import plot_model
from sklearn import preprocessing
from sklearn.externals import joblib
import os
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.metrics import Recall, Precision

def standardisation(df):
    target = df.iloc[:, 3:4]
    input = df.iloc[:, 0:3]
    names = input.columns
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(input)
    scaled_df = pd.DataFrame(scaled_df, columns=names)
    scaled_df = pd.concat([scaled_df, target], axis=1)
    joblib.dump(scaler, "data/scaler.save")
    return scaled_df


def load_data():
    data = pd.read_table('data/haberman.txt', sep=',', header=0)
    df = pd.DataFrame(data)
    return df

def split_data(df, tar_oneHot):
    target = df['Status']
    y = tar_oneHot
    x = df.iloc[:, 0:3]
    x_train, x_test, y_train, y_test = train_test_split(x, y,train_size = 0.7, test_size = 0.3, random_state =42)
    return (x_train, y_train), (x_test, y_test), x, y, target


def oneHot_encode(df):
    encoder = ce.OneHotEncoder(cols=['Status'])
    tar_oneHot = encoder.fit_transform(df['Status'])
    dfd = pd.concat([df, tar_oneHot], axis=1)
    return dfd, tar_oneHot


def save_file(df):
    print("\n",df.head(),"\n")
    np.savetxt(r'data/habermanEncoded.txt', df.values, delimiter=',', fmt='%s')


def create_model():
    # Creation du model
    print('******************************')
    model = Sequential()
    # Empilement des couches de neurones
    # Couche-1
    model.add(Dense(256, input_dim=3, kernel_initializer='uniform', activation='tanh'))
    # Couche-2
    model.add(Dense(128, activation='tanh'))
    # Couche-3
    model.add(Dense(64, activation='tanh'))
    # Couche-4
    model.add(Dense(32, activation='tanh'))
    # Couche-5
    model.add(Dense(16, activation='tanh'))
    # Couche-6
    model.add(Dense(4, activation='tanh'))
    # Couche-7
    model.add(Dense(2, activation='tanh'))

    # Descente de gradient stochastique "Adam"
    Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)

    # Compilation du model -- Fonction d'apprentissage --
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy', Recall(), Precision()])
    return model


def ploting_model(model):
    plt.subplot(211)
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('Model\'s performance')
    plt.ylabel('Loss')
    plt.legend(['train', 'test'], loc='upper left')
    plt.subplot(212)
    plt.plot(model.history['accuracy'])
    plt.plot(model.history['val_accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def load_scaler(scaler_filename):
    scaler = joblib.load(scaler_filename)
    # x_test_scaled = scaler.transform(x_test)
    return scaler


if __name__ == '__main__':

    # Chargement de données "DataSet"
    df = load_data()

    # Standardisation
    scaled_df = standardisation(df)

    # Encodage de l'attribut class "Target"
    # Encodage Binaire
    dfd, tar_oneHot = oneHot_encode(scaled_df)

    # Subdivision des données en 2 groupes
    # 1.    Données d'apprentissage
    # 2.    Données de teste et validation
    (x_train, y_train), (x_test, y_test), x, y, target = split_data(dfd, tar_oneHot)

    print('\n====================================\n')
    print(type(x_train),'\t',type(y_train),'\t',type(y_train),'\t',type(y_test))
    print('\n====================================\n')

    # Enregistrement de DataSet après l'encodage
    save_file(dfd)

    tensorboard = TensorBoard()

    # Initialiser le model
    model = create_model()
    history = History()

    # Arrêt de l'apprentissage lorsque la précésion augemante
    # L'apprentissage s'arrete lorsque l'une des conditions suivantes a été vérifiée
    # 1.    min_delta : erreur minimale
    # 2.    patiente : nombre d'itérations minimum
    estimator = EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=200, verbose=1, mode='auto')

    # la fonction objectif
    fitnes = model.fit(x,y,validation_data=(x_test,y_test),callbacks=[estimator, history, tensorboard],verbose=2 ,epochs=10000)

    # Nombre à virgule flottante
    np.set_printoptions(suppress=True)

    # Prediction
    predict = model.predict(x_test)
    print("\n******************************** Predicted : ********************************\n", predict[0:10] ,"\n******************************** Target : ******************************** \n", y_test[0:10] ,"\n")

    # La valeur de l'Erreur "Loss Error"
    print("\nLoss Error : ",mean_squared_error(y_test,predict),"\n")

    model.save('data/model_file.h5')
    weights = model.get_weights()
    np.savetxt('data/weights.csv', weights, fmt='%s', delimiter=',')

    # Plot model's performance "LOSS" & "ACCURACY"
    ploting_model(history)
