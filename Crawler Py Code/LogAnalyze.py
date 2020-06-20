#-*- coding:utf-8 -*-
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.abspath(__file__))
csvPath = os.path.dirname(os.path.dirname(path))
CrawledPath = os.path.join(csvPath, 'CrawledData')
nameList = os.listdir(CrawledPath)
nameList.sort()
crawledFileName = nameList[-1]
crawledData = pd.read_csv(os.path.join(CrawledPath, crawledFileName), \
    dtype = {'종목코드':np.str, '종목명':np.str, \
    '현재가':np.int64, '시가총액':np.int64})

codes = crawledData['종목코드'].to_list()

logFile = open('Run_MakeAllModel.py.log', 'r')
lines = logFile.readlines()

max_loss = 0.0
max_val_loss = 0.0
min_loss = 1000000.0
min_val_loss = 1000000.0
sum_loss = 0.0
sum_val_loss = 0.0

sec_max_loss = 0.0
sec_max_val_loss = 0.0
sec_min_loss = 1000000.0
sec_min_val_loss = 1000000.0
sec_sum_loss = 0.0
sec_sum_val_loss = 0.0

checkLine = 'Epoch 100/100'
one_check = '.h5 Done'
sec_check = '_win50_sec10.h5 Done.'

losses = []
val_losses = []
sec_losses = []
sec_val_losses = []

index = 0
stock_num = 0
# 한줄 씩 읽어서
# Epoch 100/100이 나오면
# 다음에 나오는 h5 파일이 *.h5인지 *_win50_sec10.h5인지 검사
# loss와 val_loss 값을 맞는 쪽에 저장.
for line in tqdm(lines):
    if checkLine in line:
        line = lines[index + 1]
        # loss 값과 val_loss 값을 걸러냄
        splited = line.split('- loss: ')
        splited = splited[1].split(' - val_loss: ')
        loss = float(splited[0])
        val_loss = float(splited[1])
        
        for h5Line in lines[index + 2:]:
            if sec_check in h5Line:
                if h5Line.split(sec_check)[0] in codes:
                    sec_sum_loss += loss
                    sec_sum_val_loss += val_loss
                    
                    # 최대, 최소값 갱신
                    if sec_max_loss < loss:
                        sec_max_loss = loss
                    elif sec_min_loss > loss:
                        sec_min_loss = loss
                    if sec_max_val_loss < val_loss:
                        sec_max_val_loss = val_loss
                    elif sec_min_val_loss > val_loss:
                        sec_min_val_loss = val_loss
                    
                    sec_losses.append(loss)
                    sec_val_losses.append(val_loss)
                lines.remove(h5Line)
                break

            elif one_check in h5Line:
                if h5Line.split(one_check)[0] in codes:
                    sum_loss += loss
                    sum_val_loss += val_loss
                    
                    # 최대, 최소값 갱신
                    if max_loss < loss:
                        max_loss = loss
                    elif min_loss > loss:
                        min_loss = loss
                    if max_val_loss < val_loss:
                        max_val_loss = val_loss
                    elif min_val_loss > val_loss:
                        min_val_loss = val_loss
                    
                    losses.append(loss)
                    val_losses.append(val_loss)

                    stock_num += 1
                lines.remove(h5Line)
                break
    index += 1

stock_num = float(stock_num)

sum_loss = round(sum_loss / stock_num, 6)
sum_val_loss = round(sum_val_loss / stock_num, 6)

sec_sum_loss = round(sec_sum_loss / stock_num, 6)
sec_sum_val_loss = round(sec_sum_val_loss / stock_num, 6)