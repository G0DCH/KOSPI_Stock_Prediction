#-*- coding:utf-8 -*-

import datetime
from LSTM.Run_MakeAllPredict import MakeAllPredict
from LSTM.SaveToDB.SaveToDB import SaveToDB

date = datetime.date.today()
date = date.strftime('%Y%m%d')

print
print('*********************************')
print(date + " : Update DB Start")

MakeAllPredict(50, 10)
SaveToDB()

print(date + ' : Update DB Finished')
print('*********************************')
print
