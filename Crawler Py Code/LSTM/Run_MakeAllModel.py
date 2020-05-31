#-*- coding:utf-8 -*-

from Run_AppendSection import *
from multiprocessing import Process

def MakeAllModel(window_Size, sectionLength):
    start = time.time()
    print('MakeAllModel Start')
    PriceChangeDirName = 'PriceChangedData'

    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.dirname(path)

    PriceChangePath = os.path.join(path, PriceChangeDirName)

    dataFileNameList = os.listdir(PriceChangePath)
    dataFileNameList.sort()

    i = 0

    threadList = []
    threadArgNameList = []
    for dataFileName in dataFileNameList:
        p = Process(target=AppendMakeModel, args=(window_Size, dataFileName, sectionLength))
        p.daemon = True
        if len(threadList) < 10:
            p.start()
            threadList.append(p)
            threadArgNameList.append(dataFileName)
        else:
            p2 = threadList.pop(0)
            p2ArgName = threadArgNameList.pop(0)
            p2.join()
            i += 1
            print("{}/{} : {}".format(i, len(dataFileNameList), p2ArgName))
            p.start()
            threadList.append(p)
            threadArgNameList.append(dataFileName)

    for p in threadList:
        p.join()
        tArgName = threadArgNameList.pop(0)
        i += 1
        print("{}/{} : {}".format(i, len(dataFileNameList), tArgName))

    print('MakeAllModel Finished')
    print('MakeAllModel time : ' + str(timedelta(seconds = time.time() - start)))

if __name__ == "__main__":
    MakeAllModel(50, 10)