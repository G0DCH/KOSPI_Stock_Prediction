#-*- coding:utf-8 -*-

import threading
from update import update_crawl
from update import update_Foreigner_invest_crawl
from update import update_NPS_invest_crawl

update_crawl.update_crawl()

print('KOSPI Crawling done.')
print('Foreigner & NPS Crawling Start')

foreign = threading.Thread(target=update_Foreigner_invest_crawl.update_Foreigner_invest_crawl)
nps = threading.Thread(target=update_NPS_invest_crawl.update_NPS_invest_crawl)

foreign.start()
nps.start()

foreign.join()
nps.join()

print('Crawling Finished')