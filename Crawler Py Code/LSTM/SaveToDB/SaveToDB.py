#-*- coding:utf-8 -*-

import pymysql
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# DBInfo.py 파일을 작성한 뒤 사용
# 파일은 다음과 같은 내용으로 작성해야 한다.
# host = 'db 주소'
# user = '계정 이름'
# password = '패스워드'
# charset = '문자열 포맷'
# db = 'db 이름'
import DBInfo

NAME = 'Name'
PRICE = 'Price'
PREDICT = 'Predict'

conn = pymysql.connect(host = DBInfo.host,
                    user = DBInfo.user,
                    password = DBInfo.password,
                    charset = DBInfo.charset,
                    local_infile = 1)

def MakeDB():
    try:
        with conn.cursor() as cursor:
            sql = 'CREATE DATABASE {}'.format(DBInfo.db)
            cursor.execute(sql)
        conn.commit()
    finally:
        pass

def InitTable(tableType):

    try:
        with conn.cursor() as cursor:
            # db에 넣을 형태를 만들어 csv 파일로 저장
            tmpFileName = 'tmp.csv'
            path = os.path.dirname(os.path.abspath(__file__))
            csvPath = os.path.dirname(os.path.dirname(path))
            PriceChangePath = os.path.join(csvPath, 'PriceChangedData')
            CrawledPath = os.path.join(csvPath, 'CrawledData')
            predictPath = os.path.join(csvPath, 'PredictCSV')
            nameList = os.listdir(CrawledPath)
            nameList.sort()
            crawledFileName = nameList[-1]
            crawledData = pd.read_csv(os.path.join(CrawledPath, crawledFileName), \
                dtype = {'종목코드':np.str, '종목명':np.str, \
                '현재가':np.int64, '시가총액':np.int64})

            def UploadCSV(tableType):
                # csv 파일을 db에 업로드
                sql = "LOAD DATA LOCAL INFILE '{}' \
                    INTO TABLE {} CHARACTER SET utf8mb4 \
                    FIELDS TERMINATED BY ',' \
                    LINES TERMINATED BY '\n';".format(os.path.join(path, tmpFileName), tableType)
                cursor.execute(sql)
            
            if tableType == NAME:
                data = crawledData.loc[:, ['종목코드', '종목명']]
                data.to_csv(os.path.join(path, tmpFileName), header = False, index = False)

                # csv 파일을 db에 업로드
                UploadCSV(tableType)

            elif tableType == PRICE:
                for code in tqdm(crawledData['종목코드']):
                    fileName = code + '.csv'
                    data = pd.read_csv(os.path.join(PriceChangePath, fileName), \
                        dtype = {'날짜':np.int64, '종목코드':np.str, '종목명':np.str, \
                        '현재가':np.int64, '시가총액':np.int64, '외인순매수거래량':np.int64, \
                        '외인순매수거래대금':np.int64, '연기금순매수거래량':np.int64, '연기금순매수거래대금':np.int64})

                    data = data.loc[:, ['종목코드', '날짜', '현재가']]
                    data.to_csv(os.path.join(path, tmpFileName), header = False, index = False)

                    # csv 파일을 db에 업로드
                    UploadCSV(tableType)
            elif tableType == PREDICT:
                for code in tqdm(crawledData['종목코드']):
                    fileName = 'Predict_{}.csv'.format(code)
                    try:
                        import shutil
                        shutil.copyfile(os.path.join(predictPath, fileName), os.path.join(os.path.join(path, tmpFileName)))
                        UploadCSV(tableType)
                    except IOError as e:
                        print(e)
                    #data = pd.read_csv(os.path.join(predictPath, fileName),\
                    #    dtype = {'종목코드':np.str, '날짜':np.int64, '예측가':np.int64})
                    
                    #data.to_csv(os.path.join(path, tmpFileName), header = False, index = False)
            else:
                raise ValueError('{} is not correct table Name'.format(tableType))
            os.remove(os.path.join(path, tmpFileName))
        conn.commit()
    except:
        print('Init Table Failed')
    finally:
        pass

# 테이블 생성
def MakeTable(tableType):
    try:
        with conn.cursor() as cursor:
            sql = "SHOW TABLES LIKE '{}'".format(tableType)
            cursor.execute(sql)
            result = cursor.fetchall()

            if len(result) > 0:
                sql = 'DROP TABLE {}'.format(tableType)
                cursor.execute(sql)
                result = cursor.fetchall()

            if tableType == NAME:
                sql = 'CREATE TABLE {} (\
                        Code char(6) NOT NULL, \
                        Name varchar(30) NOT NULL) \
                        default charset=utf8mb4'.format(tableType)
                cursor.execute(sql)
            elif tableType == PRICE:
                sql = 'CREATE TABLE {} (\
                    Code char(6) NOT NULL, \
                    MarketDate date NOT NULL, \
                    Price int NOT NULL) \
                    default charset=utf8mb4'.format(tableType)
                cursor.execute(sql)
            elif tableType == PREDICT:
                sql = 'CREATE TABLE {} (\
                    Code char(6) NOT NULL, \
                    MarketDate date NOT NULL, \
                    PredictPrice int NOT NULL) \
                    default charset=utf8mb4'.format(tableType)
                cursor.execute(sql)
            else:
                raise ValueError('{} is not correct table Name'.format(tableType))
        conn.commit()
    except:
        print('Make Table Failed')
    finally:
        pass

def Init():
    try:
        with conn.cursor() as cursor:
            sql = 'USE {}'.format(DBInfo.db)
            cursor.execute(sql)
            cursor.fetchall()
    except:
        MakeDB()
    finally:
        try:
            with conn.cursor() as cursor:
                sql = 'SELECT DATABASE()'
                cursor.execute(sql)
                result = cursor.fetchall()
                print(result)
        finally:
                pass

def CloseDB():
    conn.close()
    print('DB Closed')