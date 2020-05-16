#-*- coding:utf-8 -*-

import time
import warnings
import os
import sys
import numpy as np
import pandas as pd
from datetime import timedelta
from tqdm import tqdm
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt

start = time.time()

column = ['종목코드', '현재가', '외인순매수거래량', '외인순매수거래대금', '연기금순매수거래량', '연기금순매수거래대금']
#column = ['현재가', '외인순매수거래량', '외인순매수거래대금', '연기금순매수거래량', '연기금순매수거래대금']
emptyFrame = pd.DataFrame(columns = column)

pivotDatas = []

# 입력 받은 데이터를 정규화함
def Normalize(dataList, stockCode):
    normalizedDatas = []

    start = time.time()
    print('Normalize Start')
    global pivotDatas

    for window in tqdm(dataList):
        normalizedWindow = window.copy()
        pivot = window.copy()
        pivotDatas.append(pivot.iloc[0, 1])
        for i in range(len(pivot)):
            pivot.iloc[i] = pivot.iloc[0]
        normalizedWindow.loc[:] = window.loc[:] / pivot[:] - 1
        normalizedWindow['종목코드'] = stockCode
        normalizedWindow = normalizedWindow[column]
        print(normalizedWindow)
        normalizedDatas.append(normalizedWindow.values.tolist())

    result = np.array(normalizedDatas)

    print('Normalize Finished')
    print('Normalized time : ' + str(timedelta(seconds = time.time() - start)))

    return result


# 학습 데이터 셋 만듬
def LoadData(window_Size):
    
    start = time.time()
    print('Load Data Start')

    PriceChangeDirName = 'PriceChangedData'

    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.dirname(path)

    PriceChangePath = os.path.join(path, PriceChangeDirName)
    dataFileNameList = os.listdir(PriceChangePath)

    result = []
    # 리스트에 window_Size 동안의 데이터를 추가함
    #for dataFileName in dataFileNameList:
    #    data = pd.read_csv(os.path.join(PriceChangePath, dataFileName))
    stockCode = '005930.csv'.split('.')[0]
    data = pd.read_csv(os.path.join(PriceChangePath, stockCode), \
        dtype = {'날짜':np.int64, '종목코드':np.str, '종목명':np.str, \
            '현재가':np.int64, '시가총액':np.int64, '외인순매수거래량':np.int64, \
            '외인순매수거래대금':np.int64, '연기금순매수거래량':np.int64, '연기금순매수거래대금':np.int64})

    data = data.loc[:, column[1:]]
    windowSize = window_Size + 1
    
    for index in range(len(data) - windowSize + 1):
        stockData = data[index : index + windowSize].copy()
        result.append(stockData)

    stockData = data[-50:].copy()
    stockData = Normalize([stockData], stockCode)
    result = Normalize(result, stockCode)

    # 90퍼센트는 train용 10퍼센트는 validate용으로 씀
    row = int(round(result.shape[0] * 0.9))
    train = result[:row, :]
    np.random.shuffle(train)

    x_train = train[:, :-1]
    y_train = train[:, -1, 1]
    x_test = result[row:, :-1]
    x_test = np.append(x_test, stockData, axis = 0)
    y_test = result[row:, -1, 1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    print('LoadData Finished')
    print('LoadData time : ' + str(timedelta(seconds = time.time() - start)))

    return [x_train, y_train, x_test, y_test]

# 학습 모델 생성
def BuildModel():
    model = Sequential()

    model.add(LSTM(50, input_shape=(50, 6), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop')

    return model

def Run():
    fileName = 'stock_batch512_epoch100.h5'

    x_train, y_train, x_test, y_test = LoadData(50)
    model = BuildModel()
    
    model.fit(x_train, y_train, batch_size=512, epochs=100, validation_split=0.05)
    model.save(fileName)

    #from keras.models import load_model

    #model = load_model(fileName)

    dateLength = 11
    x_test2 = x_test[-(dateLength + 1):]
    y_test2 = (y_test[-dateLength:].astype(np.float64) + 1) * pivotDatas[-dateLength:]
    pred = model.predict(x_test2)
    result_predict = []
    for i in range(-len(pred), 0):
        result_predict.append((pred[i] + 1) * pivotDatas[i])
    print(result_predict[-1])
    plt.figure(facecolor = 'white')
    plt.plot(y_test2, label='actual')
    plt.plot(result_predict, label='prediction')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    Run()
    print('Run time : ' + str(timedelta(seconds = time.time() - start)))