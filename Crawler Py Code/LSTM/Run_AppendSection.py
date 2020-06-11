#-*- coding:utf-8 -*-

from Run import *
from Run_SectionPrediction import fitModel

def AppendRun(window_Size, codeFileName, sectionLength):

    path = os.path.dirname(os.path.abspath(__file__))
    onePath = os.path.join(path, 'OneDayPredict')
    sectionPath = os.path.join(path, 'SectionPredict')

    fileName = '{}.h5'.format(codeFileName.split('.')[0])
    appendFileName = "{}_win{}_sec{}.h5".format(codeFileName.split('.')[0], window_Size, sectionLength)

    x_train0, y_train0, x_test0, y_test0 = LoadData(50, codeFileName)

    #x_train = nanToZero(x_train0, True)
    #y_train = nanToZero(y_train0, False)
    x_test = nanToZero(x_test0, True)
    y_test = nanToZero(y_test0, False)
    pivotDatas0 = nanToZero(np.array(pivotDatas), False)

    #model = BuildModel()

    #model.fit(x_train, y_train, batch_size=512, epochs=100, validation_split=0.05, verbose = 2)
    #model.save(fileName)

    from keras.models import load_model

    model = load_model(os.path.join(onePath, fileName))
    append_model = load_model(os.path.join(sectionPath, appendFileName))

    dateLength = 10
    tmpData = pd.read_csv(os.path.join(os.path.dirname(path), 'PriceChangedData', codeFileName))
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

def AppendMakeModel(window_Size, codeFileName, sectionLength):
    path = os.path.dirname(os.path.abspath(__file__))
    onePath = os.path.join(path, 'OneDayPredict')
    sectionPath = os.path.join(path, 'SectionPredict')

    if os.path.isdir(onePath) == False:
        print("No Directory : " + onePath)
        print("Make Directory : " + onePath)
        os.makedirs(onePath)

    if os.path.isdir(sectionPath) == False:
        print("No Directory : " + sectionPath)
        print("Make Directory : " + sectionPath)
        os.makedirs(sectionPath)
    fileName = '{}.h5'.format(codeFileName.split('.')[0])
    appendFileName = "{}_win{}_sec{}.h5".format(codeFileName.split('.')[0], window_Size, sectionLength)

    x_train0, y_train0, x_test0, y_test0 = LoadData(50, codeFileName)

    noneCheck = (type(x_train0) == None) or (type(y_train0) == None) or \
                (type(x_test0) == None) or (type(y_test0) == None)

    # 너무 짧아서 정규화가 안된 경우
    if noneCheck:
        return

    x_train = nanToZero(x_train0, True)
    y_train = nanToZero(y_train0, False)
    x_test = nanToZero(x_test0, True)
    y_test = nanToZero(y_test0, False)
    pivotDatas0 = nanToZero(np.array(pivotDatas), False)

    model = BuildModel()

    model.fit(x_train, y_train, batch_size=512, epochs=100, validation_split=0.05, verbose = 2)
    model.save(os.path.join(onePath, fileName))

    print('{} Done'.format(fileName))
    #model = load_model(fileName)

    if os.path.isfile(os.path.join(sectionPath, appendFileName)) == False:
        fitModel(window_Size, codeFileName, sectionLength)

if __name__ == "__main__":
    AppendRun(50, '005930.csv', 10)
    print('Run time : ' + str(timedelta(seconds = time.time() - start)))