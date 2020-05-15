#-*- coding:utf-8 -*-

import pandas as pd
import os
import sys
import time
import ReconstructCrawledData
from datetime import timedelta

def UpdateReconstructCrawledData():
    start = time.time()
    funcName = sys._getframe().f_code.co_name

    UnifyDirName = 'UnifiedCrawledData'
    ReconstructDirName = 'ReconstructedCrawledData'
    LastDateFileName = 'LastReconstructDate.txt'

    path = os.path.dirname(os.path.abspath(__file__))

    # 마지막 재구성 날짜 구함
    LastDateFilePath = os.path.join(path, LastDateFileName)

    if os.path.isfile(LastDateFilePath) == False:
        print('No Date File')
        print("Excute 'ReconstructCrawledData.py'")
        ReconstructCrawledData.ReconstructCrawledData()
        return

    LastDateFile = open(LastDateFilePath, 'r')
    lastDate = LastDateFile.read()
    LastDateFile.close()

    path = os.path.dirname(path)
    RecontructPath = os.path.join(path, ReconstructDirName)
    UnifyPath = os.path.join(path, UnifyDirName)

    dataFileNameList = os.listdir(UnifyPath)
    dataFileNameList.sort()
    lastIndex = dataFileNameList.index(lastDate + '.csv')

    dataDictionary = {}
    emptyFrame = pd.DataFrame(columns = ['날짜', '종목코드', '종목명', '현재가', '시가총액', '외인순매수거래량', 
                                        '외인순매수거래대금', '연기금순매수거래량', '연기금순매수거래대금'])

    # dataFileName은 날짜.csv임
    # 종목 코드별로 데이터를 저장함
    for index in range(lastIndex + 1, len(dataFileNameList)):
        dataFileName = dataFileNameList[index]
        UnifiedData = pd.read_csv(os.path.join(UnifyPath, dataFileName), dtype = {'종목코드':np.str})

        stockCodes = UnifiedData['종목코드']
        Date = dataFileName.split('.')[0]

        # 종목 코드 별로 데이터 저장
        for stockCode in stockCodes:
            checkData = pd.read_csv(os.path.join(RecontructPath, (stockCode + '.csv')), dtype = {'종목코드':np.str})

            if (checkData['날짜'] == int(Date)).any():
                continue

            if dataDictionary.has_key(stockCode) == False:
                #print('No Value in Dictionary, Make Key Value Pair : ' + stockCode)
                dataDictionary[stockCode] = emptyFrame
            stockData = UnifiedData[stockCodes == stockCode].copy()
            stockData['날짜'] = Date
            dataDictionary[stockCode] = dataDictionary[stockCode].append(stockData, ignore_index = True, sort = False)
        
        print(funcName + ' : ' + Date)

    print(funcName + ' : ' + 'Reconstuct DataFrame Finished')

    # 데이터 프레임을 종목 코드를 이름으로 해서 파일로 저장
    for dataKey in dataDictionary:
        data = dataDictionary[dataKey]
        fileName = str(dataKey) + '.csv'
        data.to_csv(os.path.join(RecontructPath, fileName), mode = 'a', header = False, index = False)
        #print(funcName + ' : ' + fileName)

    newDate = dataFileNameList[len(dataFileNameList) - 1].split('.')[0]
    LastDateFile = open(LastDateFilePath, 'w')
    LastDateFile.write(newDate)
    LastDateFile.close()

    print('Recontruct Data Finished')
    print('time : ' + str(timedelta(seconds = time.time() - start)))

if __name__ == "__main__":
    UpdateReconstructCrawledData()