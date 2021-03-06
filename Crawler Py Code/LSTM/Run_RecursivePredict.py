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

emptyFrame = pd.DataFrame(columns = ['현재가', '외인순매수거래량', 
                                    '외인순매수거래대금', '연기금순매수거래량', '연기금순매수거래대금'])

pivotDatas = []

def nanToZero(array, isTwo):
    tmpArray = array.copy()
    if isTwo:
        for i in tqdm(range(tmpArray.shape[0])):
            for j in range(tmpArray.shape[1]):
                tmp = tmpArray[i, j, :].astype('float64')
                tmpArray[i, j, np.isnan(tmp)] = 0
                tmpArray[i, j, np.isinf(tmp)] = 0
    else:
        tmpArray = tmpArray.astype('float64')
        tmpArray[np.isnan(tmpArray)] = 0
        tmpArray[np.isinf(tmpArray)] = 0

    return tmpArray

# 입력 받은 데이터를 정규화함
def Normalize(dataList):
    normalizedDatas = []

    start = time.time()
    print('Normalize Start')
    global pivotDatas

    for window in tqdm(dataList):
        normalizedWindow = window.copy()
        pivot = window.copy()
        pivotDatas.append(pivot.iloc[0])
        for i in range(len(pivot)):
            pivot.iloc[i] = pivot.iloc[0]
        normalizedWindow.loc[:] = window.loc[:] / pivot[:] - 1
        normalizedDatas.append(normalizedWindow.values.tolist())

    result = np.array(normalizedDatas)

    print('Normalize Finished')
    print('Normalized time : ' + str(timedelta(seconds = time.time() - start)))

    return result


# 학습 데이터 셋 만듬
def LoadData(window_Size, fileName):
    
    start = time.time()
    print('Load Data Start')

    PriceChangeDirName = 'PriceChangedData'

    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.dirname(path)

    PriceChangePath = os.path.join(path, PriceChangeDirName)

    result = []
    # 리스트에 window_Size 동안의 데이터를 추가함
    stockCode = fileName.split('.')[0]
    data = pd.read_csv(os.path.join(PriceChangePath, fileName), \
        dtype = {'날짜':np.int64, '종목코드':np.str, '종목명':np.str, \
            '현재가':np.int64, '시가총액':np.int64, '외인순매수거래량':np.int64, \
            '외인순매수거래대금':np.int64, '연기금순매수거래량':np.int64, '연기금순매수거래대금':np.int64})

    data = data.loc[:, ['현재가', '외인순매수거래량', 
        '외인순매수거래대금', '연기금순매수거래량', '연기금순매수거래대금']]
    windowSize = window_Size + 1
    
    for index in range(len(data) - windowSize + 1):
        stockData = data[index : index + windowSize].copy()
        result.append(stockData)

    stockData = data[-50:].copy()
    stockData = Normalize([stockData])
    result = Normalize(result)

    # 90퍼센트는 train용 10퍼센트는 validate용으로 씀
    row = int(round(result.shape[0] * 0.9))
    train = result[:row, :]
    np.random.shuffle(train)

    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[row:, :-1]
    x_test = np.append(x_test, stockData, axis = 0)
    y_test = result[row:, -1]

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

    model.add(Dense(5))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop')

    return model

def Run():
    codeFileName = '005930.csv'
    window_Size = 50
    sectionLength = 10

    fileName = 'Recursive_{}.h5'.format(codeFileName.split('.')[0])

    x_train0, y_train0, x_test0, y_test0 = LoadData(window_Size, codeFileName)

    x_train = nanToZero(x_train0, True)
    y_train = nanToZero(y_train0, False)
    x_test = nanToZero(x_test0, True)
    y_test = nanToZero(y_test0, False)
    pivotDatas0 = nanToZero(np.array(pivotDatas), False)

    model = BuildModel()
    
    model.fit(x_train, y_train, batch_size=512, epochs=100, validation_split=0.05, verbose=2)
    model.save(fileName)

    #from keras.models import load_model

    #model = load_model(fileName)

    dateLength = 10
    tmpData = pd.read_csv(os.path.join('/home/chlee/KOSPI_Prediction/PriceChangedData', codeFileName))
    tmpDate = tmpData['날짜']
    tmpDate = tmpDate[-dateLength:].values

    import datetime
    date = []
    for i in range(tmpDate.shape[0]):
        date.append(np.datetime64(datetime.datetime.strptime(str(tmpDate[i]), "%Y%m%d"), 'D'))
        
    for i in range(sectionLength):
        date.append(np.datetime64(date[-1].astype(datetime.datetime) + datetime.timedelta(days = 1), 'D'))
        
    date = np.array(date)

    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    x_test2 = x_test[-(dateLength + 1):-1]
    y_test2 = (y_test[-dateLength:, 0].astype(np.float64) + 1) * pivotDatas0[-dateLength:, 0]

    for i in range(sectionLength):
        pred = model.predict(x_test2)
        tmp = np.append(x_test2[-1, 1:], pred[-1])
        x_test2 = np.append(x_test2, tmp)

    pred = model.predict(x_test2)
    result_predict = []
    for i in range(-len(pred), -dateLength):
        result_predict.append(int((pred[i, 0] + 1) * pivotDatas0[i - dateLength, 0]))

    for i in range(-dateLength, 0):
        result_predict.append(int((pred[i, 0] + 1) * pivotDatas0[-1, 0]))

    print('다음 10일 간 예측가 : ' + str(result_predict[-sectionLength:, 0]))
    print('오늘 종가 : ' + str(y_test2[-1, 0]))
    plt.figure(facecolor = 'white')
    plt.plot(date[:-sectionLength], y_test2, label='actual')
    plt.plot(date, result_predict, label='prediction')
    plt.xticks(rotation = -45)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    Run()
    print('Run time : ' + str(timedelta(seconds = time.time() - start)))