#-*- coding:utf-8 -*-

import threading
import datetime
import UnifyCrawledData
from update import update_crawl
from update import update_Foreigner_invest_crawl
from update import update_NPS_invest_crawl
from reconstruct import UpdateReconstructCrawledData
import PriceChange

date = datetime.date.today()
date = date.strftime('%Y%m%d')

print
print('*********************************')
print(date + " : Update Crawling Start")

update_crawl.update_crawl()

print('KOSPI Crawling done.')
print('Foreigner & NPS Crawling Start')

foreign = threading.Thread(target=update_Foreigner_invest_crawl.update_Foreigner_invest_crawl)
nps = threading.Thread(target=update_NPS_invest_crawl.update_NPS_invest_crawl)

foreign.start()
nps.start()

foreign.join()
nps.join()

UnifyCrawledData.UnifyCrawledData()
UpdateReconstructCrawledData.UpdateReconstructCrawledData()

PriceChange.PriceChange()

print(date + ' : Update Crawling Finished')
print('*********************************')
print
