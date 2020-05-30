#-*- coding:utf-8 -*-

from Run import *
from Run_SectionPrediction import fitModel

def AppendRun():
    codeFileName = '051900.csv'
    window_Size = 50
    sectionLength = 10

    fileName = '{}.h5'.format(codeFileName.split('.')[0])
    appendFileName = "{}_win{}_sec{}.h5".format(codeFileName.split('.')[0], window_Size, sectionLength)

    x_train0, y_train0, x_test0, y_test0 = LoadData(50, codeFileName)

    x_train = nanToZero(x_train0, True)
    y_train = nanToZero(y_train0, False)
    x_test = nanToZero(x_test0, True)
    y_test = nanToZero(y_test0, False)
    pivotDatas0 = nanToZero(np.array(pivotDatas), False)

    model = BuildModel()

    model.fit(x_train, y_train, batch_size=512, epochs=100, validation_split=0.05, verbose = 2)
    model.save(fileName)

    #from keras.models import load_model

    #model = load_model(fileName)
    #append_model = load_model(appendFileName)

    dateLength = 10
    tmpData = pd.read_csv(os.path.join('/home/chlee/KOSPI_Prediction/PriceChangedData', codeFileName))
    tmpDate = tmpData['날짜']
    tmpDate = tmpDate[-dateLength:].values

    import datetime
    tmp = []
    for i in range(tmpDate.shape[0]):
        tmp.append(np.datetime64(datetime.datetime.strptime(str(tmpDate[i]), "%Y%m%d"), 'D'))
        
    for i in range(sectionLength):
        tmp.append(np.datetime64(tmp[-1].astype(datetime.datetime) + datetime.timedelta(days = 1), 'D'))
        
    tmp = np.array(tmp)

    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    x_test2 = x_test[-(dateLength + 1):-1]
    y_test2 = (y_test[-dateLength:].astype(np.float64) + 1) * pivotDatas[-dateLength:]
    append_x_test = x_test[-2:]
    pred = model.predict(x_test2)
    pred2 = append_model.predict(append_x_test)
    pred2 = pred2[-1]
    #print(pred2)
    result_predict = []
    for i in range(-len(pred), 0):
        result_predict.append(int((pred[i] + 1) * pivotDatas0[i - 1]))

    for pred2Data in pred2:
        result_predict.append(int((pred2Data + 1) * pivotDatas0[-1]))
    print('다음 10일 간 예측가 : ' + str(result_predict[-sectionLength:]))
    print('오늘 종가 : ' + str(y_test2[-1]))
    plt.figure(facecolor = 'white')
    plt.plot(tmp[:-sectionLength], y_test2, label='actual')
    plt.plot(tmp, result_predict, label='prediction')
    plt.xticks(rotation = -45)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    Run()
    print('Run time : ' + str(timedelta(seconds = time.time() - start)))