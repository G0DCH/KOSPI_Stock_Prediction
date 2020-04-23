#-*- coding:utf-8 -*-

import pandas as pd
import os
import sys

def ReconstructCrawledData():
    funcName = sys._getframe().f_code.co_name

    UnifyDirName = 'UnifiedCrawledData'
    ReconstructDirName = 'ReconstructedCrawledData'

    path = os.path.dirname(os.path.abspath(__file__))

    RecontructPath = os.path.join(path, ReconstructDirName)
    UnifyPath = os.path.join(path, UnifyDirName)

    if os.path.isdir(RecontructPath) == False:
        print("No Directory : " + RecontructPath)
        print("Make Directory : " + RecontructPath)
        os.makedirs(os.path.join(RecontructPath))

    dataFileNameList = os.listdir(UnifyPath)
    dataFileNameList.sort()

    dataDictionary = {}
    emptyFrame = pd.DataFrame(columns = ['날짜', '종목코드', '종목명', '현재가', '시가총액', '외인순매수거래량', 
                                        '외인순매수거래대금', '연기금순매수거래량', '연기금순매수거래대금'])

    # dataFileName은 날짜.csv임
    # 종목 코드별로 데이터를 저장함
    for dataFileName in dataFileNameList:
        UnifiedData = pd.read_csv(os.path.join(UnifyPath, dataFileName))

        stockCodes = UnifiedData['종목코드']

        # 종목 코드 별로 데이터 저장
        for stockCode in stockCodes:
            if dataDictionary.has_key(stockCode) == False:
                print('No Value in Dictionary, Make Key Value Pair : ' + stockCode)
                dataDictionary[stockCode] = emptyFrame
            stockData = UnifiedData[stockCodes == stockCode].copy()
            Date = dataFileName.split('.')[0]
            stockData['날짜'] = Date
            dataDictionary[stockCode] = dataDictionary[stockCode].append(stockData, ignore_index = True, sort = False)
        
        print(funcName + ' : ' + Date)

    print(funcName + ' : ' + 'Reconstuct DataFrame Finished')

    # 데이터 프레임을 종목 코드를 이름으로 해서 파일로 저장
    for dataKey in dataDictionary:
        data = dataDictionary[dataKey]
        data.to_csv(os.path.join(RecontructPath, dataKey), header = True, index = False)
        print(funcName + ' : ' + dataKey)

    print('Recontruct Data Finished')

if __name__ == "__main__":
    ReconstructCrawledData()