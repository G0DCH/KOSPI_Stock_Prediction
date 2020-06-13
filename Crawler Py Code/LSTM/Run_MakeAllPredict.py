#-*- coding:utf-8 -*-

from Run_AppendSection import *
from multiprocessing import Process

def MakeAllPredict(window_Size, sectionLength):
    start = time.time()
    print('MakeAllPredict Start')

    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.dirname(path)
    CrawledPath = os.path.join(path, 'CrawledData')

    dataFileNameList = os.listdir(CrawledPath)
    dataFileNameList.sort()

    crawledFileName = dataFileNameList[-1]

    crawledData = pd.read_csv(os.path.join(CrawledPath, crawledFileName), \
        dtype = {'종목코드':np.str, '종목명':np.str, \
        '현재가':np.int64, '시가총액':np.int64})

    i = 0

    processList = []
    processArgNameList = []
    for code in crawledData['종목코드']:
        dataFileName = code + '.csv'
        p = Process(target=MakeCSV, args=(window_Size, dataFileName, sectionLength))
        p.daemon = True
        if len(processList) < 100:
            p.start()
            processList.append(p)
            processArgNameList.append(dataFileName)
        else:
            p2 = processList.pop(0)
            p2ArgName = processArgNameList.pop(0)
            p2.join()
            i += 1
            print("{}/{} : {}".format(i, crawledData.shape[0], p2ArgName))
            p.start()
            processList.append(p)
            processArgNameList.append(dataFileName)

    for p in processList:
        p.join()
        tArgName = processArgNameList.pop(0)
        i += 1
        print("{}/{} : {}".format(i, crawledData.shape[0], tArgName))

    print('MakeAllPredict Finished')
    print('MakeAllPredict time : ' + str(timedelta(seconds = time.time() - start)))

if __name__ == "__main__":
    MakeAllPredict(50, 10)