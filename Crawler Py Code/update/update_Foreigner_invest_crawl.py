#-*- coding:utf-8 -*-

# 오늘부터 2001년 1월 1일 까지의 외인 매매 동향을 크롤링 해옴
import pandas as pd
import numpy as np
import requests
from io	import BytesIO
import datetime
import os
import sys

def update_Foreigner_invest_crawl():
	time = datetime.date(2001,1,1)
	dateDiff = datetime.date.today() - time
	dateDiff = dateDiff.days
	time = datetime.date.today()
	funcName = sys._getframe().f_code.co_name

	dirName = "ForeignerCrawledData"
	KOSPIdirName = "CrawledData/"
	path = os.path.dirname(os.path.abspath(__file__))
	path = os.path.split(path)
	KOSPIpath = path[0] + '/' + KOSPIdirName
	path = path[0] + '/' + dirName
	
	if os.path.isdir(path) == False:
		print("No Directory : " + path)
		print("Make Directory : " + path)
		os.makedirs(os.path.join(path))

	for i in range(dateDiff):
		date = time.strftime('%Y%m%d')
		time += datetime.timedelta(days=-1)

		if os.path.isfile(KOSPIpath + date + '.csv') == False:
			continue

		# 크롤링 진행 중 과거에 크롤링 한 데이터가 있으면 크롤링 중단
		if os.path.isfile(path + '/' + date + '.csv') == True:
			break

		headers = {'Referer' : 'http://marketdata.krx.co.kr/mdi',
					'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36'}

		gen_otp_url = "http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx"
		gen_otp_data = {'name' : "fileDown",
						'filetype' : "csv",
						'url' : "MKD/04/0404/04040400/mkd04040400",
						'stctype' : "STK",
						'var_invr_cd' : "9000",
						'schdate' : date,
						'etctype' : 'ST',
						'pagePath' : "/contents/MKD/04/0404/04040400/MKD04040400.jsp"}
		otp = requests.post(gen_otp_url, gen_otp_data, headers=headers)
		code = otp.content

		down_url = "http://file.krx.co.kr/download.jspx"
		down_data = {'code' : code,}
		down = requests.post(down_url, down_data, headers=headers)
		down.encoding = 'utf-8-sig'

		if len(down.content) > 1000:
			down = pd.read_csv(BytesIO(down.content), header=0, thousands=',')
			down = down.loc[:, ['종목코드', '종목명', '순매수거래량','순매수거래대금']]
			down.to_csv(path + '/' + date + '.csv', header=True, index=False)
			print(funcName + ' : ' + date)

	print("Update Foreign Crawling Finished")

if __name__ == "__main__":
	update_Foreigner_invest_crawl()