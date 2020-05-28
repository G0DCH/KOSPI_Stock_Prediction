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

codeTable = {}

def Hash(code):
    p = 31
    m = 1e9 + 9
    
    hashValue = 0
    pow_p = 1

    for i in range(len(code)):
        hashValue = (hashValue + (ord(code[i]) - ord('0') + 1) * pow_p) % m
        pow_p = (p * pow_p) % m
    codeTable[hashValue] = code

    return hashValue

pivotCode = Hash('000010')

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
def Normalize(dataList, stockCode):
    normalizedDatas = []

    start = time.time()

    if stockCode == '005930':
        global pivotDatas
        for window in dataList: #tqdm(dataList):
            normalizedWindow = window.copy()
            pivot = window.copy()
            pivotDatas.append(pivot.iloc[0, 0])
            for i in range(len(pivot)):
                pivot.iloc[i] = pivot.iloc[0]
            normalizedWindow.loc[:] = window.loc[:] / pivot[:] - 1
            normalizedWindow['종목코드'] = stockCode
            normalizedWindow = normalizedWindow[column]
            normalizedDatas.append(normalizedWindow.values.tolist())
    else:
        for window in dataList: #tqdm(dataList):
            normalizedWindow = window.copy()
            pivot = window.copy()
            #pivotDatas.append(pivot.iloc[0, 0])
            for i in range(len(pivot)):
                pivot.iloc[i] = pivot.iloc[0]
            normalizedWindow.loc[:] = window.loc[:] / pivot[:] - 1
            normalizedWindow['종목코드'] = stockCode
            normalizedWindow = normalizedWindow[column]
            normalizedDatas.append(normalizedWindow.values.tolist())

    result = np.array(normalizedDatas)

    return result


# 학습 데이터 셋 만듬
def LoadData(window_Size):
    
    start = time.time()
    print('Load Data Start')

    PriceChangeDirName = 'PriceChangedData'
    dirName = 'NPYAllStockCode'
    path = os.path.dirname(os.path.abspath(__file__))

    tmpSavePath = os.path.join(path, dirName)
    tmpFileName = 'tmpData'
    path = os.path.dirname(path)

    if os.path.isdir(tmpSavePath) == False:
        os.makedirs(tmpSavePath)

    PriceChangePath = os.path.join(path, PriceChangeDirName)
    dataFileNameList = os.listdir(PriceChangePath)
    dataFileNameList.sort()

    train = None
    #x_test = np.zeros((0, 0))
    #y_test = np.zeros((0, 0))
    # 리스트에 window_Size 동안의 데이터를 추가함
    i = 1
    length = len(dataFileNameList)

    # 1400번째 부터 이어 받기
    train = np.load('/home/chlee/KOSPI_Prediction/LSTM/NPYAllStockCode/tmpData_1400.npy')
    for dataFileName in dataFileNameList:
        # 1400번째 까지는 패스
        if i < 1401:
            i += 1
            continue
        windowResult = []
        stockCode = dataFileName.split('.')[0]
        data = pd.read_csv(os.path.join(PriceChangePath, dataFileName), \
            dtype = {'날짜':np.int64, '종목코드':np.str, '종목명':np.str, \
                '현재가':np.int64, '시가총액':np.int64, '외인순매수거래량':np.int64, \
                '외인순매수거래대금':np.int64, '연기금순매수거래량':np.int64, '연기금순매수거래대금':np.int64})

        data = data.loc[:, column[1:]]
        windowSize = window_Size + 1
        
        for index in range(len(data) - windowSize + 1):
            stockData = data[index : index + windowSize].copy()
            windowResult.append(stockData)

        result = Normalize(windowResult, stockCode)
        row = int(round(result.shape[0] * 0.9))
        if type(train) == type(None):
            train = result[:row, :]
        else:
            # 51일 미만의 주가 데이터는 거름
            if (result.shape[0] == 0) == False:
                train = np.append(train, result[:row, :], axis = 0)
        print('%d/%d : %s done' % (i, length, dataFileName))

        # 임시 저장
        if i % 100 == 0:
            tmp_File_Name = tmpFileName + '_' + str(i)
            np.save(os.path.join(tmpSavePath, tmp_File_Name), train)
        i += 1
        #x_test = np.append(x_test, result[row:, :-1], axis = 0)
        #y_test = np.append(y_test, result[row:, -1, 1], axis = 0)
        
    np.save(os.path.join(tmpSavePath, 'tmpData_Finished'), train)
    for index_0 in range(train.shape[0]):
        for index_1 in range(train.shape[1]):
            train[index_0, index_1, 0] = \
                float(Hash(train[index_0, index_1, 0])) / float(pivotCode) - 1
    # 90퍼센트는 train용 10퍼센트는 validate용으로 씀
    np.random.shuffle(train)

    x_train = train[:, :-1]
    y_train = train[:, -1, 1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    #x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    print('LoadData Finished')
    print('LoadData time : ' + str(timedelta(seconds = time.time() - start)))

    return [x_train, y_train] #, x_test, y_test]

def LoadTestData(window_Size, fileName):
    start = time.time()
    print('Load Test Data Start')

    PriceChangeDirName = 'PriceChangedData'

    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.dirname(path)

    PriceChangePath = os.path.join(path, PriceChangeDirName)

    stockCode = fileName.split('.')[0]
    data = pd.read_csv(os.path.join(PriceChangePath, fileName), \
        dtype = {'날짜':np.int64, '종목코드':np.str, '종목명':np.str, \
            '현재가':np.int64, '시가총액':np.int64, '외인순매수거래량':np.int64, \
            '외인순매수거래대금':np.int64, '연기금순매수거래량':np.int64, '연기금순매수거래대금':np.int64})

    data = data.loc[:, column[1:]]
    windowSize = window_Size + 1
    
    result = []
    for index in range(len(data) - windowSize + 1):
        stockData = data[index : index + windowSize].copy()
        result.append(stockData)

    stockData = data[-50:].copy()
    stockData = Normalize([stockData], stockCode)
    result = Normalize(result, stockCode)

    row = int(round(result.shape[0] * 0.9))
    x_test = result[row:, :-1]
    x_test = np.append(x_test, stockData, axis = 0)
    y_test = result[row:, -1, 1]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    print('LoadTestData Finished')
    print('LoadTestData time : ' + str(timedelta(seconds = time.time() - start)))

    return [x_test, y_test]

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
    fileName = 'all_stock_batch512_epoch100.h5'

    #x_train, y_train = LoadData(50) #, x_validate, y_validate = LoadData(50)
    #x_test, y_test = LoadTestData(50, '005930.csv')

    dirName = 'NPYAllStockCode'
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, dirName)
    
    x_train_name = os.path.join(path, 'all_stock_X_Train')
    y_train_name = os.path.join(path, 'all_stock_Y_Train')
    x_test_name = os.path.join(path, 'all_stock_X_Test')
    y_test_name = os.path.join(path, 'all_stock_Y_Test')
    pivot_name = os.path.join(path, 'all_stock_pivot')

    if os.path.isdir(path) == False:
        os.makedirs(path)
    
    x_train = np.load(x_train_name + '.npy')
    y_train = np.load(y_train_name + '.npy')
    x_test = np.load(x_test_name + '.npy')
    y_test = np.load(y_test_name + '.npy')
    pivotDatas = np.load(pivot_name + '.npy')

    """
    # Nan 값이나 inf 값을 0으로 바꿈
    x_train = nanToZero(x_train, True)
    y_train = nanToZero(y_train, False)
    x_test = nanToZero(x_test, True)
    y_test = nanToZero(y_test, False)
    pivotDatas = nanToZero(pivotDatas, False)
    
    np.save(x_train_name, x_train)
    np.save(y_train_name, y_train)
    np.save(x_test_name, x_test)
    np.save(y_test_name, y_test)
    np.save(pivot_name, np.array(pivotDatas))
    """

    model = BuildModel()
    
    model.fit(x_train, y_train, batch_size=512, epochs=100, validation_split=0.05, verbose=2)
    model.save(fileName)

    #from keras.models import load_model

    #model = load_model(fileName)

    """
    dateLength = 11
    tmpData = pd.read_csv('/home/chlee/KOSPI_Prediction/PriceChangedData/005930.csv')
    tmpDate = tmpData['날짜']
    tmpDate = tmpDate[-dateLength:].values
    tmpDate = np.append(tmpDate, [tmpDate[-1] + 1])

    import datetime
    tmp = []
    for i in range(tmpDate.shape[0]):
        tmp.append(np.datetime64(datetime.datetime.strptime(str(tmpDate[i]), "%Y%m%d"), 'D'))
        
    tmp = np.array(tmp)

    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    x_test2 = x_test[-(dateLength + 1):]
    y_test2 = (y_test[-dateLength:].astype(np.float64) + 1) * pivotDatas[-dateLength:]
    pred = model.predict(x_test2)
    result_predict = []
    for i in range(-len(pred), 0):
        result_predict.append((pred[i] + 1) * pivotDatas[i])
    print(result_predict[-1])
    plt.figure(facecolor = 'white')
    plt.plot(tmp[:-1], y_test2, label='actual')
    plt.plot(tmp, result_predict, label='prediction')
    plt.xticks(rotation = -45)
    plt.legend()
    plt.show()
    """

if __name__ == "__main__":
    Run()
    print('Run time : ' + str(timedelta(seconds = time.time() - start)))