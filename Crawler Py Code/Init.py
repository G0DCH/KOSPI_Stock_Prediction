#-*- coding:utf-8 -*-

import threading
from init import init_crawl
from init import init_Foreigner_invest_crawl
from init import init_NPS_invest_crawl

init_crawl.init_crawl()

print('KOSPI Crawling done.')
print('Foreigner & NPS Crawling Start')

foreign = threading.Thread(target=init_Foreigner_invest_crawl.init_Foreigner_invest_crawl)
nps = threading.Thread(target=init_NPS_invest_crawl.init_NPS_invest_crawl)

foreign.start()
nps.start()

print('Crawling Finished')