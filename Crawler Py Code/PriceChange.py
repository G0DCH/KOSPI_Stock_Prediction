#-*- coding:utf-8 -*-

import pandas as pd
import os
import sys
import shutil
import numpy as np

def PriceChange():
    funcName = sys._getframe().f_code.co_name

    ReconstructDirName = 'ReconstructedCrawledData'
    PriceChangeFileName = 'PriceChange.csv'
    PriceChangeDirName = 'PriceChangedData'

    path = os.path.dirname(os.path.abspath(__file__))
    RecontructPath = os.path.join(path, ReconstructDirName)
    PriceChangeFileName = os.path.join(path, PriceChangeFileName)
    PriceChangePath = os.path.join(path, PriceChangeDirName)

    if(os.path.isdir(PriceChangePath)):
        print('Exist Directory : ' + PriceChangePath)
        print('Remove Directory : ' + PriceChangePath)
        shutil.rmtree(PriceChangePath)
    
    shutil.copytree(RecontructPath, PriceChangePath)

    PriceChangeFile = pd.read_csv(PriceChangeFileName, \
        dtype = {'종목코드':np.str, '현재가':np.int64, '외인순매수거래량':np.int64, '연기금순매수거래량':np.int64})

    changeFrame = ['현재가', '외인순매수거래량', '연기금순매수거래량']

    # ChangeData는 액면 변경 데이터
    # changeFile은 액면 변경을 적용할 파일
    for i in range(len(PriceChangeFile)):
        ChangeData = PriceChangeFile.iloc[i]
        changeFilePath = os.path.join(PriceChangePath, ChangeData['종목코드'] + '.csv')
        changeFile = pd.read_csv(changeFilePath, dtype = {'종목코드':np.str})
        changeFile.loc[changeFile['날짜'] <= ChangeData['날짜'], changeFrame] = \
            (changeFile[changeFile['날짜'] <= ChangeData['날짜']][changeFrame] * ChangeData['비율']).astype('int')
        changeFile.to_csv(changeFilePath, header = True, index = False)
        print(str(ChangeData['날짜']) + ' : ' + ChangeData['종목코드'])
    
    print('Price Change Finished')
    

if __name__ == "__main__":
    PriceChange()