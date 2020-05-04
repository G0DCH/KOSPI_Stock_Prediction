#-*- coding:utf-8 -*-

import time
import warnings
import os
import sys
import numpy as np
import pandas as pd
from datetime import timedelta
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

start = time.time()

#emptyFrame = pd.DataFrame(columns = ['종목코드', '현재가', '외인순매수거래량', 
#                                    '외인순매수거래대금', '연기금순매수거래량', '연기금순매수거래대금'])
emptyFrame = pd.DataFrame(columns = ['현재가', '외인순매수거래량', 
                                    '외인순매수거래대금', '연기금순매수거래량', '연기금순매수거래대금'])

pivotDatas = []

# 입력 받은 데이터를 정규화함
def Normalize(dataList):
    normalizedDatas = []

    start = time.time()
    print('Normalize Start')
    global pivotDatas

    for window in dataList:
        normalizedWindow = window.copy()
        pivot = window.copy()
        pivotDatas.append(pivot.iloc[0, 0])
        for i in range(len(pivot)):
            pivot.iloc[i] = pivot.iloc[0]
        normalizedWindow.loc[:] = window.loc[:] / pivot[:] - 1
        normalizedDatas.append(normalizedWindow.values.tolist())

    result = np.array(normalizedDatas)

    print('Normalize Finished')
    print('Normalized time : ' + str(timedelta(seconds = time.time() - start)))

    return result


# 학습 데이터 셋 만듬
def LoadData(window_Size):
    
    start = time.time()
    print('Load Data Start')

    ReconstructDirName = 'ReconstructedCrawledData'

    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.dirname(path)

    ReconstructPath = os.path.join(path, ReconstructDirName)
    dataFileNameList = os.listdir(ReconstructPath)

    result = []
    # 리스트에 window_Size 동안의 데이터를 추가함
    #for dataFileName in dataFileNameList:
    #    data = pd.read_csv(os.path.join(ReconstructPath, dataFileName))
    data = pd.read_csv(os.path.join(ReconstructPath, '005930.csv'))

    """
    scaler = MinMaxScaler()
    scale_cols = ['현재가', '외인순매수거래량', '외인순매수거래대금', '연기금순매수거래량', '연기금순매수거래대금']

    df_scaled = scaler.fit_transform(data[scale_cols])

    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = scale_cols
    """
    
    data = data.loc[:, ['현재가', '외인순매수거래량', 
        '외인순매수거래대금', '연기금순매수거래량', '연기금순매수거래대금']]
    windowSize = window_Size + 1
    
    for index in range(len(data) - windowSize):
    #for index in range(len(data) - windowSize - 12, len(data) - windowSize):
        stockData = data[index : index + windowSize].copy()
        result.append(stockData)

    stockData = data[-50:].copy()
    stockData = Normalize([stockData])
    result = Normalize(result)

    """
    row = int(round(df_scaled.shape[0] * 0.9))
    train = df_scaled[:-row]
    test = df_scaled[-row:]

    feature_cols = ['현재가', '외인순매수거래량', '외인순매수거래대금', '연기금순매수거래량', '연기금순매수거래대금']
    label_cols = ['현재가']

    train_feature = train[feature_cols]
    train_label = train[label_cols]

    test_feature = test[feature_cols]
    test_label = test[label_cols]

    # train dataset
    train_feature, train_label = make_dataset(train_feature, train_label, window_Size)

    from sklearn.model_selection import train_test_split
    x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)

    test_feature, test_label = make_dataset(test_feature, test_label, window_Size)

    return [x_train, x_valid, y_train, y_valid, test_feature, test_label]
    """
    # 90퍼센트는 train용 10퍼센트는 validate용으로 씀
    row = int(round(result.shape[0] * 0.9))
    train = result[:row, :]
    np.random.shuffle(train)

    x_train = train[:, :-1]
    y_train = train[:, -1, 0]
    x_test = result[row:, :-1]
    x_test = np.append(x_test, stockData, axis = 0)
    y_test = result[row:, -1, 0]
    #x_test = result[:, :-1]
    #y_test = result[:, -1, 0]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    print('LoadData Finished')
    print('LoadData time : ' + str(timedelta(seconds = time.time() - start)))

    return [x_train, y_train, x_test, y_test]

# 학습 모델 생성
def BuildModel():
    model = Sequential()

    model.add(LSTM(50, input_shape=(50,5), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(output_dim=1))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop')

    """
    model = Sequential()
    
    model.add(LSTM(16, 
               input_shape=(50, 5), 
               activation='relu', 
               return_sequences=False)
          )
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    """
    """
    model.add(LSTM(50, return_sequences = True, input_shape = (50, 6)))
    model.add(LSTM(64, return_sequences = False))
    model.add(Dense(1, activation = 'linear'))
    model.compile(loss="mse", optimizer="rmsprop")
    """

    return model

"""
def make_dataset(data, label, window_size=50):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)
"""
def Run():

    #x_train, x_valid, y_train, y_valid, test_feature, test_label = LoadData(50)

    x_train, y_train, x_test, y_test = LoadData(50)
    #model = BuildModel()
    """
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp_checkpoint.h5')
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')


    history = model.fit(x_train, y_train, 
                    epochs=200, 
                    batch_size=16,
                    validation_data=(x_valid, y_valid), 
                    callbacks=[early_stop, checkpoint])

    model.load_weights(filename)
    pred = model.predict(test_feature)
    """
    
    #model.fit(x_train, y_train, batch_size=512, nb_epoch=100, validation_split=0.05)
    #model.save('batch512_epoch100.h5')

    from keras.models import load_model

    model = load_model('batch512_epoch100.h5')

    dateLength = 11
    x_test2 = x_test[-(dateLength + 1):]
    y_test2 = (y_test[-dateLength:] + 1) * pivotDatas[-dateLength:]
    pred = model.predict(x_test2)
    result_predict = []
    for i in range(-len(pred), 0):
        result_predict.append((pred[i] + 1) * pivotDatas[i])
    plt.figure(facecolor = 'white')
    plt.plot(y_test2, label='actual')
    plt.plot(result_predict, label='prediction')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    Run()
    print('Run time : ' + str(timedelta(seconds = time.time() - start)))