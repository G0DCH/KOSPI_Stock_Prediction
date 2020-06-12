#-*- coding:utf-8 -*-

import pymysql
import os
import pandas as pd
import numpy as np

# DBInfo.py 파일을 작성한 뒤 사용
# 파일은 다음과 같은 내용으로 작성해야 한다.
# host = 'db 주소'
# user = '계정 이름'
# password = '패스워드'
# charset = '문자열 포맷'
# db = 'db 이름'
import DBInfo

conn = pymysql.connect(host = DBInfo.host,
                    user = DBInfo.user,
                    password = DBInfo.password,
                    charset = DBInfo.charset,
                    local_infile = 1)

def MakeDB(conn):
    try:
        with conn.cursor() as cursor:
            sql = 'CREATE DATABASE {}'.format(DBInfo.db)
            cursor.execute(sql)
        conn.commit()
    finally:
        pass

def InitTable(conn, code):
    try:
        with conn.cursor() as cursor:
            # db에 넣을 형태를 만들어 csv 파일로 저장
            path = os.path.dirname(os.path.abspath(__file__))
            PriceChangePath = os.path.join(os.path.dirname(os.path.dirname(path)), 'PriceChangedData')
            fileName = code + '.csv'
            data = pd.read_csv(os.path.join(PriceChangePath, fileName), \
                dtype = {'날짜':np.int64, '종목코드':np.str, '종목명':np.str, \
                '현재가':np.int64, '시가총액':np.int64, '외인순매수거래량':np.int64, \
                '외인순매수거래대금':np.int64, '연기금순매수거래량':np.int64, '연기금순매수거래대금':np.int64})

            data = data.loc[:, ['날짜', '현재가']]
            tmpFileName = 'tmp.csv'
            data.to_csv(os.path.join(path, tmpFileName), header = False, index = False)
            
            # csv 파일을 db에 업로드
            sql = "LOAD DATA LOCAL INFILE '{}' \
                INTO TABLE {} CHARACTER SET utf8mb4 \
                FIELDS TERMINATED BY ',' \
                LINES TERMINATED BY '\n';".format(os.path.join(path, tmpFileName), 'Price_{}'.format(code))
            cursor.execute(sql)
            os.remove(os.path.join(path, tmpFileName))
        conn.commit()
    finally:
        pass

def MakeTable(conn, code):
    try:
        with conn.cursor() as cursor:
            sql = 'CREATE TABLE {} (\
                    MarketDate date, \
                    Price int) default charset=utf8mb4'.format('Price_{}'.format(code))
            cursor.execute(sql)
        conn.commit()
    except:
        print('Make Table Failed')
    finally:
        InitTable(conn, code)
        print('Make {} Table Done'.format(code))
        pass

def Init():
    try:
        with conn.cursor() as cursor:
            sql = 'USE {}'.format(DBInfo.db)
            cursor.execute(sql)
            cursor.fetchall()
    except:
        MakeDB(conn)
    finally:
        try:
            with conn.cursor() as cursor:
                sql = 'SELECT DATABASE()'
                cursor.execute(sql)
                result = cursor.fetchall()
                print(result)
        finally:
                pass

        try:
            MakeTable(conn, '005930')
        finally:
            sql = 'select * from {}'.format('Price_{}'.format('005930'))
            result = pd.read_sql_query(sql, conn)
            print(result)
            pass
        conn.close()