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
        for i in range(tmpArray.shape[0]):
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

    #start = time.time()
    #print('Normalize Start')
    #global pivotDatas

    for window in dataList:
        normalizedWindow = window.copy()
        pivot = window.copy()
        #pivotDatas.append(pivot.iloc[0, 0])
        for i in range(len(pivot)):
            pivot.iloc[i] = pivot.iloc[0]
        normalizedWindow.loc[:] = window.loc[:] / pivot[:] - 1
        normalizedDatas.append(normalizedWindow.values.tolist())

    result = np.array(normalizedDatas)

    #print('Normalize Finished')
    #print('Normalized time : ' + str(timedelta(seconds = time.time() - start)))

    return result


# 학습 데이터 셋 만듬
def LoadData(window_Size, fileName, sectionLength):
    
    start = time.time()
    print('{} Load Data Start'.format(fileName))

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
    windowSize = window_Size + sectionLength
    
    for index in range(len(data) - windowSize + 1):
        stockData = data[index : index + windowSize].copy()
        result.append(stockData)

    testStockData = []
    # 리스트에 예측용 데이터만 담음.
    for index in range(len(data) - windowSize + 1, len(data) - window_Size + 1):
        stockData = data[index : index + window_Size].copy()
        testStockData.append(stockData)

    result = Normalize(result)
    testStockData = Normalize(testStockData)

    # 너무 짧아서 정규화가 안된 경우
    if (result.shape[0] == 0) == True:
        print('Short!!!!! {} LoadData Finished'.format(fileName))
        print('{} LoadData time : {}'.format(fileName, str(timedelta(seconds = time.time() - start))))
        return [None, None, None, None]

    # 90퍼센트는 train용 10퍼센트는 validate용으로 씀
    row = int(round(result.shape[0] * 0.9))
    train = result[:row, :]
    np.random.shuffle(train)

    x_train = train[:, :-sectionLength]
    y_train = train[:, -sectionLength:, 0]
    x_test = result[row:, :-sectionLength]
    x_test = np.append(x_test, testStockData, axis = 0)
    y_test = result[row:, -sectionLength:, 0]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    print('{} LoadData Finished'.format(fileName))
    print('{} LoadData time : {}'.format(fileName, str(timedelta(seconds = time.time() - start))))

    return [x_train, y_train, x_test, y_test]

# 학습 모델 생성
def BuildModel():
    model = Sequential()

    model.add(LSTM(50, input_shape=(50,5), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop')

    return model

def Run():
    fileName = 'Section_batch512_epoch100.h5'

    codeFileName = '005930.csv'

    x_train0, y_train0, x_test0, y_test0 = LoadData(50, codeFileName, 10)

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
    tmp = []
    for i in range(tmpDate.shape[0]):
        tmp.append(np.datetime64(datetime.datetime.strptime(str(tmpDate[i]), "%Y%m%d"), 'D'))
        
    tmp = np.array(tmp)

    y_tmp = y_test[:-1, 0].copy()
    y_tmp2 = (y_tmp.astype(np.float64) + 1) * pivotDatas0[-(len(y_tmp) + 11):-11]
    y_tmp = y_test[-1]
    y_tmp2 = np.append(y_tmp2, (y_tmp.astype(np.float64) + 1) * pivotDatas0[-11])

    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    x_test2 = x_test[-(dateLength):]
    y_test2 = y_tmp2[-dateLength:]
    pred = model.predict(x_test2)
    result_predict = []
    for i in range(-len(pred), 0):
        result_predict.append((pred[i] + 1) * pivotDatas0[i])
    plt.figure(facecolor = 'white')
    plt.plot(tmp, y_test2, label='actual')
    print(result_predict[-1])
    print(y_test2[-1])
    for i in range(len(result_predict)):
        plt.plot(tmp[:10], result_predict[i])
        tmp = np.append(tmp[1:], np.datetime64(tmp[-1].astype(datetime.datetime) + datetime.timedelta(days = 1), 'D'))
    plt.xticks(rotation = -45)
    plt.legend()
    plt.show()

def fitModel(window_Size, codeFileName, sectionLength):
    path = os.path.dirname(os.path.abspath(__file__))
    sectionPath = os.path.join(path, 'SectionPredict')

    fileName = "{}_win{}_sec{}.h5".format(codeFileName.split('.')[0], window_Size, sectionLength)

    x_train0, y_train0, x_test0, y_test0 = LoadData(50, codeFileName, 10)

    noneCheck = (type(x_train0) == None) or (type(y_train0) == None) or \
            (type(x_test0) == None) or (type(y_test0) == None)

    # 너무 짧아서 정규화가 안된 경우
    if noneCheck:
        return

    x_train = nanToZero(x_train0, True)
    y_train = nanToZero(y_train0, False)
    x_test = nanToZero(x_test0, True)
    y_test = nanToZero(y_test0, False)
    pivotDatas0 = nanToZero(np.array(pivotDatas), False)

    model = BuildModel()
    
    model.fit(x_train, y_train, batch_size=512, epochs=100, validation_split=0.05, verbose=2)
    model.save(os.path.join(sectionPath, fileName))

    print("{} Done.".format(fileName))

    return model

if __name__ == "__main__":
    Run()
    print('Run time : ' + str(timedelta(seconds = time.time() - start)))