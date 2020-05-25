#-*- coding:utf-8 -*-

import threading
import datetime
import UnifyCrawledData
from init import init_crawl
from init import init_Foreigner_invest_crawl
from init import init_NPS_invest_crawl
from reconstruct import ReconstructCrawledData
import PriceChange

date = datetime.date.today()
date = date.strftime('%Y%m%d')

print
print('*********************************')
print(date + " : Init Crawling Start")

init_crawl.init_crawl()

print('KOSPI Crawling done.')
print('Foreigner & NPS Crawling Start')

foreign = threading.Thread(target=init_Foreigner_invest_crawl.init_Foreigner_invest_crawl)
nps = threading.Thread(target=init_NPS_invest_crawl.init_NPS_invest_crawl)

foreign.start()
nps.start()

foreign.join()
nps.join()

UnifyCrawledData.UnifyCrawledData()
ReconstructCrawledData.ReconstructCrawledData()

PriceChange.PriceChange()

print(date + ' : Init Crawling Finished')
print('*********************************')
print
