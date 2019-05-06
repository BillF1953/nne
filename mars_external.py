# /home/tylerf/tensorflow/bin/python3.5

import os
import logging
import os
import pickle
import pandas
from multiprocessing import Pool
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import keras
import csv


class mars():   
    
    def __init__(self, sample_size=1000):      
        self.sample_size = sample_size
        self.x_min = 0
        self.x_max = 1
        self.x_dim = 5
        self.y_dim = 1
        
    def simulate(self):
        x = np.random.uniform(self.x_min, self.x_max, size=(self.sample_size, 5))     
        y = 10*np.sin(np.pi*x[:,0]*x[:,1]) - 20*(x[:,2]-0.05)**2 + 10*x[:,3] + 5*x[:,4] + np.random.normal(0, 10, self.sample_size)
        return (x,y)


def build_mars_estimator():  

    nn1 = keras.Sequential()
    # This is worth about 9% if we cut neurons down to 25
    nn1.add(keras.layers.Dense(num_lay_nodes, kernel_initializer="uniform", 
                                   activation='sigmoid', input_dim=5))
    nn1.add(keras.layers.Dense(units=1, activation='linear'))
    # training
    # sgd = keras.optimizers.SGD(lr=0.0001);
    adam = keras.optimizers.Adam(lr=lr)
    # model.compile(loss='mse', optimizer=sgd)
    nn1.compile(loss='mse', optimizer=adam)
    return(nn1)

debug = False
num_lay_nodes = 12
lr = .01
epochs = 3000
keras_verbose = 0
nn = build_mars_estimator()
x=np.array([.5,.5,.5,.5,.5]).reshape(1,5)
gr_truth=10*np.sin(np.pi*x[:,0]*x[:,1]) - 20*(x[:,2]-0.05)**2 + 10*x[:,3] + 5*x[:,4]    
    



def estimate(train):    


    ##This isn't strictly necessary but maintaisn the same intialiation 
    ## parameterization for each neural network. 
    Wsave = nn.get_weights()
    nn.fit(train[:,0:5], train[:,5], epochs=epochs,verbose=keras_verbose)

    # predictions
    # Linear predict 10 
    #x_pred = np.arange(10, 11, 1, dtype='float32').reshape(-1, 1)

    x_pred = np.array([.5,.5,.5,.5,.5])
    predictions = nn.predict(x_pred.reshape((1,5)))
    nn.set_weights(Wsave)

    if debug:
        plt.scatter(simulated_data['x'], simulated_data['y'], edgecolors='g')
        plt.plot(x_pred, predictions, 'r')
        plt.legend(['Predicted Y', 'Actual Y'])
        plt.show()

    return(predictions)
    
    
if  __name__ == '__main__':

    num_samp = 50
    start_time = datetime.now()

    x,y = mars(10000).simulate()
    regress=np.column_stack((x,y))
    size=len(regress)
    samples=[regress[np.random.choice(size, int(size**.5))] for i in range(num_samp)]



    with Pool(8) as p:
        seq=p.map(estimate, samples)
    #change type from map iter
    pred=np.fromiter(seq, dtype=np.int)


    print("This took {} seconds to run".format(datetime.now()-start_time))
    print("Ground Truth {} Mean {} Var {}".format(gr_truth, np.mean(pred),np.var(pred)))
    
    
    


