import csv
import datetime
import math
from nltk.cluster import KMeansClusterer, cosine_distance
import numpy as np
import itertools
import os
import pandas as pd
import pickle
import pymysql
import sys
import threading

import DeepMetric_GUI

# matplotlib libraries
from matplotlib.pyplot import figure
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

# sklearn libraries
from sklearn import datasets
from sklearn.cluster import estimate_bandwidth, KMeans, MeanShift
from sklearn.feature_selection import SelectFromModel
from sklearn.manifold import TSNE
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split # to split the data into two parts
from sklearn.pipeline import Pipeline # pipeline making
from sklearn.preprocessing import MinMaxScaler, StandardScaler # for normalization

# scipy libraries
from scipy.sparse import *
from scipy.stats import pearsonr, spearmanr, zscore

# Keras and Tensorflow
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping
from keras.layers import Activation, AveragePooling1D, Dropout, Dense, Flatten, LSTM, TimeDistributed
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import model_from_json, load_model, Sequential
from keras.optimizers import SGD
from keras.utils import np_utils, to_categorical


# Currently unused, kept for posterity
#from dbutils_reviewed import *
#from sklearn.cross_validation import KFold # use for cross validation
#from mpl_toolkits.mplot3d import Axes3D
#import seaborn as sns
#%matplotlib notebook
#from dbutils_copy1 import *



#foldername = "/project/ub-per/MarchData/"
data_folder = "MarchData"
folder_names = "foldernames.txt"
refined_data = "refinedMarch"
filenames = "filenames"

filelist = sorted(os.listdir(data_folder))

folder_list = []
#with open("dataframes/foldernames.txt") as p2:
with open(folder_names) as p2:
    for line in p2:
        folder_list.append(line[:-1])
print(len(folder_list))

#######################################################################################################################

def tic():
    # Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

#######################################################################################################################

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

#######################################################################################################################

def dataset_resize(dframe, num):
    table = dframe
    #print table.shape, num ,table.shape[0]-num, num, table.shape[1]
    dataX = np.zeros((table.shape[0]-num,num,table.shape[1]))
    dataY = np.zeros((table.shape[0]-num,table.shape[1]))
    for i in range(len(table)-num):#print table[i:i+num, :].shape , table[i+num-1, :].shape
        dataX[i,:,:] = table[i:i+num, :]#table[i:i+num, :]
        dataY[i,:] = table[i+num, :]#print i ,sum(sum(dataX[i,:,:]- dataX[i-1,:,:]))
    return dataX, dataY

#######################################################################################################################

def create_Keras_model():
    model = Sequential()

    model.add(LSTM(32, activation='relu', stateful=False, return_sequences=True,
                   input_shape=(32, 12)))  # 32 timesteps, and 23 features 1
    model.add(TimeDistributed(Dense(1)))
    model.add(AveragePooling1D())
    model.add(Flatten())
    model.add(Dense(12, activation='relu'))

    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    model.summary()
    return model

#######################################################################################################################

def train_model(model):
    temp = []
    tfile1 = np.array([])
    for k in range(0, 1):  # len(folder_list)):
        machine_list = []
        with open("filenames/" + folder_list[k]) as p1:
            for line in p1:
                machine_list.append(line[:-1])
        cpu_count = int(folder_list[k].split("cpu")[0])
        for ff in range(0, len(machine_list)):
            tic()
            name_machine = machine_list[ff]
            if os.path.isfile("/project/ub-per/refinedFeb/" + name_machine + ".csv") == False:
                continue
            data_frame = pd.read_csv("/project/ub-per/refinedFeb/" + name_machine + ".csv")
            data_frame = data_frame.loc[:, ~data_frame.columns.str.contains('^Unnamed')]
            data_frame = data_frame.loc[:, ~data_frame.columns.str.contains('^nfsclient.bytes.read.direct')]
            data_frame = data_frame.loc[:, ~data_frame.columns.str.contains('^nfsclient.bytes.write.direct')]
            data_frame = data_frame.loc[:, ~data_frame.columns.str.contains('^disk.dev.read_bytes')]
            data_frame = data_frame.loc[:, ~data_frame.columns.str.contains('^gpfs.fsios')]
            data_frame = data_frame.loc[:, ~data_frame.columns.str.contains('^nfsclient.bytes.read.server')]
            data_frame = data_frame.loc[:, ~data_frame.columns.str.contains(
                "^nfsclient")]  # - data_frame.mean(axis = 1 ))#/(data_frame.max(axis=1)-data_frame.min(axis=1))
            data_frame = data_frame.loc[:, ~data_frame.columns.str.contains("^kernel.percpu.cpu.nice")]
            data_frame = data_frame.loc[:, ~data_frame.columns.str.contains("^kernel.percpu.cpu.irq.hard")]
            data_frame = data_frame.iloc[:, 1:]
            # data_frame = data_frame.loc[:, data_frame.columns.str.contains('^kernel')]
            # data_frame = data_frame.iloc[:,1:]
            minval = min(data_frame.shape[0], 632)
            print(data_frame.shape)
            # if minval <= 1280:
            #    continue
            data_frame = data_frame.iloc[:minval, :]
            # print data_frame.shape
            table = data_frame.apply(zscore, axis=1)
            table1 = np.zeros((632, 12))
            if len(table) < 632:
                continue
            for i in range(632):
                table1[i, :] = (table[i])
            [dataX, datay] = dataset_resize(table1, 32)
            print(dataX.shape, datay.shape, ff)
            rsayi = np.random.randint(1, 20)
            if rsayi == 9 or rsayi == 5:
                history = model.fit(dataX, datay, batch_size=30, epochs=25,
                                    verbose=1)  # ,shuffle=False)#validation_split=0.33,
                model.reset_states()
                temp.append(ff)
                tfile1 = np.append(tfile1, history.history['loss'])
                # print(history.history['acc'])
            toc()

        model.save("models/June3_smoothModel_e100.h5")
        return model

#######################################################################################################################

#def plot_stuff():

#######################################################################################################################

def predict_anomaly(model):
    nn = 2400
    lenw = 32
    testingset = 300
    prediction = np.zeros(((testingset) - 32, nn - lenw, 12))
    prediction_diff = np.zeros(((testingset) - 32, nn - lenw, 12))
    dataylist = np.zeros(((testingset) - 32, nn - lenw, 12))
    prediction1 = np.zeros(((testingset) - 32, nn - lenw, 12))
    prediction_diff1 = np.zeros(((testingset) - 32, nn - lenw, 12))
    dataylist1 = np.zeros(((testingset) - 32, nn - lenw, 12))
    for k in range(0, 1):  # len(folder_list)):
        machine_list = []
        with open(filenames + "/" + folder_list[k]) as p1:
            for line in p1:
                machine_list.append(line[:-1])
        cpu_count = int(folder_list[k].split("cpu")[0])
        cnt = 0
        for ff in range(325):  # testing_set:
            tic()
            name_machine = machine_list[ff]
            print(ff)
            if os.path.isfile(refined_data + "/" + name_machine + ".csv") == False:
                continue
            data_frame = pd.read_csv(refined_data + "/" + name_machine + ".csv")
            data_frame = data_frame.loc[:, ~data_frame.columns.str.contains('^Unnamed')]
            data_frame = data_frame.loc[:, ~data_frame.columns.str.contains('^nfsclient.bytes.read.direct')]
            data_frame = data_frame.loc[:, ~data_frame.columns.str.contains('^nfsclient.bytes.write.direct')]
            data_frame = data_frame.loc[:, ~data_frame.columns.str.contains('^disk.dev.read_bytes')]
            data_frame = data_frame.loc[:, ~data_frame.columns.str.contains('^gpfs.fsios')]
            data_frame = data_frame.loc[:, ~data_frame.columns.str.contains('^nfsclient.bytes.read.server')]
            # data_frame = data_frame.loc[:, data_frame.columns.str.contains('^kernel')]
            data_frame = data_frame.loc[:, ~data_frame.columns.str.contains(
                "^nfsclient")]  # - data_frame.mean(axis = 1 ))#/(data_frame.max(axis=1)-data_frame.min(axis=1))
            data_frame = data_frame.loc[:, ~data_frame.columns.str.contains("^kernel.percpu.cpu.nice")]
            data_frame = data_frame.loc[:, ~data_frame.columns.str.contains("^kernel.percpu.cpu.irq.hard")]
            data_frame = data_frame.iloc[:, 1:]
            if data_frame.shape[0] < 1300:
                continue


            # data_frame = data_frame.loc[:, ~data_frame.columns.str.contains('^Unnamed')]
            # data_frame = data_frame.loc[:, data_frame.columns.str.contains('^kernel.percpu.cpu')]
            minval = min(data_frame.shape[0], 2400)
            data_frame = data_frame.iloc[:minval, :]  # +math.pow(10,-10)
            table = data_frame.apply(zscore, axis=1)

            # table = (data_frame - data_frame.mean(axis=1))/data_frame.std(ddof=0)
            if table.shape[0] == 2400:
                em = np.zeros((2400, 12))
                table1 = np.zeros((2400, 12))
                for i in range(2400):
                    em[i, :] = (table[i])
                table = em
                for u in range(2400):
                    if u > 1800 and u < 1900:
                        table1[u] = table[u] + 2
                    else:
                        table1[u] = table[u]

                [datax, datay] = dataset_resize(table, 32)
                prediction[cnt, :, :] = (model.predict(datax))
                prediction_diff[cnt, :, :] = prediction[cnt, :, :] - datay
                dataylist[cnt, :, :] = datay

                [datax1, datay1] = dataset_resize(table1, 32)
                prediction1[cnt, :, :] = (model.predict(datax1))
                prediction_diff1[cnt, :, :] = prediction1[cnt, :, :] - datay1
                dataylist1[cnt, :, :] = datay1
                cnt += 1
            toc()
            return prediction, prediction_diff, prediction1, prediction_diff1

#######################################################################################################################

def temp_manipulations(prediction, prediction_diff, prediction_1, prediction_diff1):
    temp = np.zeros((prediction_diff.shape[0], prediction_diff.shape[1]))
    temp1 = np.zeros((prediction_diff.shape[0], prediction_diff.shape[1]))
    for i in range(prediction_diff.shape[0]):
        #print(i, prediction_diff[i])
        temp[i, :] = (np.linalg.norm(prediction_diff[i], axis=1))
        temp1[i, :] = (np.linalg.norm(prediction_diff1[i], axis=1))

    for i in range(len(temp) - 1):
        if sum(np.isnan(temp[i, :])) > 0:
            temp = np.delete(temp, i, 0)
            i = i - 1

    for i in range(len(temp1) - 1):
        if sum(np.isnan(temp1[i, :])) > 0:
            temp1 = np.delete(temp1, i, 0)
            i = i - 1

    temp3 = np.zeros((272, prediction_diff.shape[1]))
    temp4 = np.zeros((272, prediction_diff.shape[1]))
    mnval = np.mean(temp[:271, :], axis=0)
    mnval1 = np.mean(temp1[:271, :], axis=0)
    for i in range(267):
        temp3[i, :] = (temp[i, :] / mnval)
        temp4[i, :] = (temp1[i, :] / mnval)

    return temp, temp1, temp3, temp4
#######################################################################################################################

if __name__ == '__main__':
    app = DeepMetric_GUI.Application()
    app.title("Deep Metric")
    #model = create_Keras_model()
    #trained_model = train_model(model)

    model = load_model("June3_smoothModel_e100.h5")
    #print(model.history.history["loss"])
    prediction, prediction_diff, prediction1, prediction_diff1 = predict_anomaly(model)
    temp1, temp2, temp3, temp4 = temp_manipulations(prediction, prediction_diff, prediction1, prediction_diff1)

    plt.ioff()
    figures = []
    canvas_x = 20
    canvas_y = 20
    for v in range(temp1.shape[0]):
        if v % 3 == 0 and v != 0:
            canvas_x = 20
            canvas_y += 320
        fig = figure(figsize=(4, 3), facecolor='w', edgecolor='k')
        plt.title("Prediction on " + str(v))
        plt.plot(temp1[v])
        plt.plot(temp2[v])
        plt.xlabel("time scale")
        plt.ylabel("prediction discrepency")
        figures.append(app.draw_figure(fig, (canvas_x, canvas_y)))
        app.update_idletasks()
        app.update()
        canvas_x += 460
        #plt.savefig("figurePred" + "%03d" % (v,) + ".pdf")
        if canvas_y + 400 > app.scroll_size:
            app.scroll_size = canvas_y + 400
            app.canvas.config(scrollregion=(0, 0, 1000, app.scroll_size))
        mpl.pyplot.close()

    threading.Thread(target=app.mainloop(), args=()).start()




