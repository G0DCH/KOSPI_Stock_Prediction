#-*- coding:utf-8 -*-

import datetime
import time
from LSTM.Run_MakeAllPredict import MakeAllPredict
from LSTM.SaveToDB.SaveToDB import SaveToDB
import os

start = time.time()
date = datetime.date.today()
date = date.strftime('%Y%m%d')

print
print('*********************************')
print(date + " : Update DB Start")

MakeAllPredict(50, 10)
SaveToDB()
path = os.path.dirname(os.path.abspath(__file__))
os.system(os.path.join(path, 'pngUpload.sh'))

print(date + ' : Update DB Finished')
print('Update DB time : ' + str(datetime.timedelta(seconds = time.time() - start)))
print('*********************************')
print
