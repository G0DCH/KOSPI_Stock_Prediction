#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys

def UnifyCrawledData():
    funcName = sys._getframe().f_code.co_name

    KOSPIDirName = 'CrawledData'
    ForeignDirName = 'ForeignerCrawledData'
    NPSDirName = 'NPSCrawledData'
    UnifyDirName = 'UnifiedCrawledData'

    path = os.path.dirname(os.path.abspath(__file__))

    KOSPIPath = os.path.join(path, KOSPIDirName)
    ForeignPath = os.path.join(path, ForeignDirName)
    NPSPath = os.path.join(path, NPSDirName)
    UnifyPath = os.path.join(path, UnifyDirName)

    if os.path.isdir(UnifyPath) == False:
        print("No Directory : " + UnifyPath)
        print("Make Directory : " + UnifyPath)
        os.makedirs(os.path.join(UnifyPath))

    dataFileNameList = os.listdir(KOSPIPath)

    for dataFileName in dataFileNameList:
        dataFilePath = os.path.join(UnifyPath, dataFileName)

        # 이전에 병합을 했다면 패스
        if os.path.isfile(dataFilePath) == True:
            continue

        KOSPIDataFile = pd.read_csv(os.path.join(KOSPIPath, dataFileName), dtype = {'종목코드':np.str})
        ForeignDataFile = pd.read_csv(os.path.join(ForeignPath, dataFileName), dtype = {'종목코드':np.str})
        NPSDataFile = pd.read_csv(os.path.join(NPSPath, dataFileName), dtype = {'종목코드':np.str})

        ForeignDataFile.rename(columns = {'순매수거래량' : '외인순매수거래량'}, inplace = True)
        ForeignDataFile.rename(columns = {'순매수거래대금' : '외인순매수거래대금'}, inplace = True)
        NPSDataFile.rename(columns = {'순매수거래량' : '연기금순매수거래량'}, inplace = True)
        NPSDataFile.rename(columns = {'순매수거래대금' : '연기금순매수거래대금'}, inplace = True)

        UnifiedData = pd.merge(KOSPIDataFile, ForeignDataFile, how = 'inner', on = ['종목코드', '종목명'])
        UnifiedData = pd.merge(UnifiedData, NPSDataFile, how = 'inner', on = ['종목코드', '종목명'])

        UnifiedData.to_csv(os.path.join(UnifyPath, dataFileName), header = True, index = False)

        print(funcName + ' : ' + dataFileName)

    print('Unify Data Finished')

if __name__ == "__main__":
    UnifyCrawledData()