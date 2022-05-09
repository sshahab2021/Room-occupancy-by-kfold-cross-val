from datetime import datetime

import keras
import numpy as np
import pandas
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import KFold, cross_val_score

def stringToDateMatrix(str):
    dt = datetime.strptime(str, "%Y-%m-%d %H:%M:%S")
    return[dt.weekday(), dt.hour, dt.minute, dt.second]

def loadData(filename):
    table = pandas.read_csv(filename, sep=',')
    dataset = table.values
    input = dataset[:, 1:6]
    raw_datetime = dataset[:, 0]
    dates = np.array([stringToDateMatrix(x) for x in raw_datetime])
    input = np.concatenate((dates, input), axis=1)
    output = dataset[:, 6]
    return (input, output)

def oneLayerModel():
    model = Sequential()
    model.add(Dense(9, input_dim=9, init='normal', activation='tanh'))
    model.add(Dense(1, init='normal', activation='tanh'))
    return {"model": model, "desc": "One layer arch."}

def twoLayerModel():
    model = Sequential()
    model.add(Dense(9, input_dim=9, init='normal', activation='tanh'))
    model.add(Dense(9, input_dim=9, init='normal', activation='tanh'))
    model.add(Dense(1, init='normal', activation='tanh'))
    return {"model": model, "desc": "Two layer arch."}

def compileSGD(model):
    model['model'].compile(loss='mean_squared_error', metrics=[
        "accuracy"], optimizer='sgd')
    model['desc'] = model['desc'] + "+SGD"
    return model

def compileAdam(model):
    model['model'].compile(loss='mean_squared_error', metrics=[
        "accuracy"], optimizer='adam')
    model['desc'] = model['desc'] + "+Adam"
    return model

def compileAdammax(model):
    opt = keras.optimizers.Adamax(
        lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model['model'].compile(loss='mean_squared_error', metrics=[
        "accuracy"], optimizer=opt)
    model['desc'] = model['desc'] + "+Adamax"
    return model

def trainModelKeras(model, X, Y):
    model['model'].fit(X, Y,
                       nb_epoch=40,
                       batch_size=40,
                       verbose=0,
                      
                       )
    return model

def evaluateModel(model):
    print ("************** ", str(model['desc']), " ******************")
    X, Y = loadData("data/datatest.txt")
    score = model['model'].evaluate(X, Y, batch_size=16)
    print(score)
    X, Y = loadData("data/datatest2.txt")
    score2 = model['model'].evaluate(X, Y, batch_size=16)
    print(score2)

X, Y = loadData("data/datatraining.txt")
evaluateModel(trainModelKeras(compileSGD(oneLayerModel()), X, Y))
evaluateModel(trainModelKeras(compileAdam(oneLayerModel()), X, Y))
evaluateModel(trainModelKeras(compileAdammax(oneLayerModel()), X, Y))
evaluateModel(trainModelKeras(compileSGD(twoLayerModel()), X, Y))
evaluateModel(trainModelKeras(compileAdam(twoLayerModel()), X, Y))
evaluateModel(trainModelKeras(compileAdammax(twoLayerModel()), X, Y))