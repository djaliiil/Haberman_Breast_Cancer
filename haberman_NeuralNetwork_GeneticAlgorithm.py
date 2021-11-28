import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, History
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
import category_encoders as ce
from sklearn import preprocessing
from keras.layers import LSTM, Input, Dense, Activation, ELU, PReLU
from keras.models import Model, Sequential
from keras.initializers import Constant
from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray

np.random.seed(1120)

# =============================================================
def load_data():
    data = pd.read_table('data/haberman.txt', sep=',', header=0)
    df = pd.DataFrame(data)
    return df
# =============================================================
def split_data(df, tar_bin):
    target = df['Status']
    y = tar_bin
    x = df.iloc[:, 0:3]
    x_train, x_test, y_train, y_test = train_test_split(x, y,train_size = 0.7, test_size = 0.3, random_state =42)
    return (x_train, y_train), (x_test, y_test), x, y, target
# =============================================================
def standardisation(df):
    target = df.iloc[:, 3:4]
    input = df.iloc[:, 0:3]
    names = input.columns
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(input)
    scaled_df = pd.DataFrame(scaled_df, columns=names)
    scaled_df = pd.concat([scaled_df, target], axis=1)
    return scaled_df
# =============================================================

def oneHot_encode(df):
    encoder = ce.OneHotEncoder(cols=['Status'])
    tar_oneHot = encoder.fit_transform(df['Status'])
    dfd = pd.concat([df, tar_oneHot], axis=1)
    return dfd, tar_oneHot
# =============================================================
# =============================================================
def train_evaluate(ga_individual_solution):
    global df, learning_rate, fct_act
    global x_train, y_train, x_test, y_test, x, y, target
    global df, dfd, tar_bin, scaled_df

    print("\n******************* Individu *********************\n")
    print(ga_individual_solution)
    print("\n**************************************************\n")
    # Decode GA solution to integer for window_size and num_units
    num_units_h1_bits = BitArray(ga_individual_solution[0:8])
    num_units_h2_bits = BitArray(ga_individual_solution[8:16])
    num_units_h3_bits = BitArray(ga_individual_solution[16:24])
    num_units_h4_bits = BitArray(ga_individual_solution[24:32])
    num_units_h5_bits = BitArray(ga_individual_solution[32:40])
    num_units_h6_bits = BitArray(ga_individual_solution[40:48])
    lr = str(BitArray(ga_individual_solution[48:51]))[2:]
    transfer_fct = str(BitArray(ga_individual_solution[51:53]))[2:]
    slope = str(BitArray(ga_individual_solution[53:55]))[2:]
    lr = learning_rate[lr]
    transfer_fct = fct_act[transfer_fct]
    slope = fct_act_slope[slope]

    num_units_h1 = num_units_h1_bits.uint
    num_units_h2 = num_units_h2_bits.uint
    num_units_h3 = num_units_h3_bits.uint
    num_units_h4 = num_units_h4_bits.uint
    num_units_h5 = num_units_h5_bits.uint
    num_units_h6 = num_units_h6_bits.uint
    perceptron = [num_units_h1, num_units_h2, num_units_h3, num_units_h4, num_units_h5, num_units_h6]

    print('\nNum of units : ', perceptron, '\n')

    model = Sequential()
    bool = False
    for i in range(0, len(perceptron)):
        if((bool == False) and (perceptron[i] != 0)):
            if(transfer_fct == 'elu'):
                ELU(alpha=slope)
                model.add(Dense(perceptron[i], input_dim=3, kernel_initializer='uniform', activation='elu'))
            elif(transfer_fct == 'prelu'):
                model.add(Dense(perceptron[i], input_dim=3, kernel_initializer='uniform', activation=PReLU(Constant(value=slope))))
            else:
                model.add(Dense(perceptron[i], input_dim=3, kernel_initializer='uniform', activation=transfer_fct))
            bool = True
        elif((bool == True) and (perceptron[i] != 0)):
            if(transfer_fct == 'elu'):
                ELU(alpha=slope)
                model.add(Dense(perceptron[i], activation='elu'))
            elif(transfer_fct == 'prelu'):
                model.add(Dense(perceptron[i], activation=PReLU(Constant(value=slope))))
            else:
                model.add(Dense(perceptron[i], activation=transfer_fct))
    if(transfer_fct == 'elu'):
        ELU(alpha=slope)
        model.add(Dense(2, activation='elu'))
    elif(transfer_fct == 'prelu'):
        model.add(Dense(2, activation=PReLU(Constant(value=slope))))
    else:
        model.add(Dense(2, activation=transfer_fct))

    Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
    model.compile(optimizer='adam',loss='mean_squared_error')
    estimator = EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=50, verbose=1, mode='auto')
    fitnes = model.fit(x,y,validation_data=(x_test,y_test),callbacks=[estimator],verbose=2 ,epochs=10000)
    np.set_printoptions(suppress=True)
    predict = model.predict(x_test)

    # Calcule de RMSE score comme fitness score Pour l'Algorithme Genetique
    rmse = np.sqrt(mean_squared_error(y_test, predict))
    print('Validation RMSE: ', rmse,'\n')

    return rmse,
# =============================================================
# =============================================================
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



if __name__ == '__main__':

    learning_rate = {
        '000': 0.001, '001': 0.002, '010': 0.005, '011': 0.01,
        '100': 0.02, '101': 0.05, '110': 0.1, '111': 0.2
    }
    fct_act = {
        '00': 'sigmoid', '01': 'tanh',
        '10': 'elu', '11': 'prelu'
    }
    fct_act_slope = {
        '00': 0.5, '01': 0.7,
        '10': 0.9, '11': 1.1
    }

    df = load_data()
    scaled_df = standardisation(df)
    dfd, tar_bin = oneHot_encode(scaled_df)
    (x_train, y_train), (x_test, y_test), x, y, target = split_data(dfd, tar_bin)


    population_size = 10
    num_generations = 5
    gene_length = 55

    # As we are trying to minimize the RMSE score, that's why using -1.0.
    # In case, when you want to maximize accuracy for instance, use 1.0
    creator.create('FitnessMax', base.Fitness, weights = (-1.0,))
    creator.create('Individual', list , fitness = creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('binary', bernoulli.rvs, 0.5)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n = gene_length)
    toolbox.register('population', tools.initRepeat, list , toolbox.individual)

    toolbox.register('mate', tools.cxOrdered)
    toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.6)
    toolbox.register('select', tools.selRoulette)
    toolbox.register('evaluate', train_evaluate)

    population = toolbox.population(n = population_size)
    r = algorithms.eaSimple(population, toolbox, cxpb = 0.4, mutpb = 0.1, ngen = num_generations, verbose = False)
    best_individuals = tools.selBest(population,k = 1)
    num_units_h1, num_units_h2, num_units_h3, num_units_h4, num_units_h5, num_units_h6 = None, None, None, None, None, None
    lr, transfer_fct = None, None

    for bi in best_individuals:
        # Decode GA solution to integer for window_size and num_units
        num_units_h1_bits = BitArray(bi[0:8])
        num_units_h2_bits = BitArray(bi[8:16])
        num_units_h3_bits = BitArray(bi[16:24])
        num_units_h4_bits = BitArray(bi[24:32])
        num_units_h5_bits = BitArray(bi[32:40])
        num_units_h6_bits = BitArray(bi[40:48])
        lr = str(BitArray(bi[48:51]))[2:]
        transfer_fct = str(BitArray(bi[51:53]))[2:]
        slope = str(BitArray(bi[53:55]))[2:]
        lr = learning_rate[lr]
        transfer_fct = fct_act[transfer_fct]
        slope = fct_act_slope[slope]

        num_units_h1 = num_units_h1_bits.uint
        num_units_h2 = num_units_h2_bits.uint
        num_units_h3 = num_units_h3_bits.uint
        num_units_h4 = num_units_h4_bits.uint
        num_units_h5 = num_units_h5_bits.uint
        num_units_h6 = num_units_h6_bits.uint
        perceptron = [num_units_h1, num_units_h2, num_units_h3, num_units_h4, num_units_h5, num_units_h6]

        print('\nNum of units : ', perceptron, '\n')

    model = Sequential()
    bool = False
    for i in range(0, len(perceptron)):
        if((bool == False) and (perceptron[i] != 0)):
            if(transfer_fct == 'elu'):
                ELU(alpha=slope)
                model.add(Dense(perceptron[i], input_dim=3, kernel_initializer='uniform', activation='elu'))
            elif(transfer_fct == 'prelu'):
                model.add(Dense(perceptron[i], input_dim=3, kernel_initializer='uniform', activation=PReLU(Constant(value=slope))))
            else:
                model.add(Dense(perceptron[i], input_dim=3, kernel_initializer='uniform', activation=transfer_fct))
            bool = True
        elif((bool == True) and (perceptron[i] != 0)):
            if(transfer_fct == 'elu'):
                ELU(alpha=slope)
                model.add(Dense(perceptron[i], activation='elu'))
            elif(transfer_fct == 'prelu'):
                model.add(Dense(perceptron[i], activation=PReLU(Constant(value=slope))))
            else:
                model.add(Dense(perceptron[i], activation=transfer_fct))
    if(transfer_fct == 'elu'):
        ELU(alpha=slope)
        model.add(Dense(2, activation='elu'))
    elif(transfer_fct == 'prelu'):
        model.add(Dense(2, activation=PReLU(Constant(value=slope))))
    else:
        model.add(Dense(2, activation=transfer_fct))

    Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
    model.compile(optimizer='adam',loss='mean_squared_error', metrics=['mse', 'accuracy'])
    history = History()
    estimator = EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=50, verbose=1, mode='auto')
    fitnes = model.fit(x,y,validation_data=(x_test,y_test),callbacks=[estimator, history],verbose=2 ,epochs=10000)
    np.set_printoptions(suppress=True)
    predict = model.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_test, predict))
    print('Test RMSE: ', rmse)
    print("\nLoss Error : ",mean_squared_error(y_test, predict),"\n")

    model.save('data/model_AG.h5')
    weights = model.get_weights()
    np.savetxt('data/weights_AG.csv', weights, fmt='%s', delimiter=',')

    ploting_model(history)
