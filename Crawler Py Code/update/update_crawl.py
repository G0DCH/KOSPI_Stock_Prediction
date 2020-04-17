#-*- coding:utf-8 -*-

# 크롤링을 현재 날짜부터 시작해서 과거 2001년 까지 진행
import pandas as pd
import numpy as np
import requests
from io	import BytesIO
import datetime
import os

def update_crawl():
	time = datetime.date(2001,1,1)
	dateDiff = datetime.date.today() - time
	dateDiff = dateDiff.days
	time = datetime.date.today()

	dirName = "CrawledData"
	if os.path.isdir(dirName) == False:
		print("No Directory : " + dirName)
		print("Make Directory : " + dirName)
		os.makedirs(os.path.join(dirName))

	for i in range(dateDiff):
		date = time.strftime('%Y%m%d')
		time += datetime.timedelta(days=-1)

		# 크롤링 진행 중 과거에 크롤링 한 데이터가 있으면 크롤링 중단
		if os.path.isfile(dirName + '/' + date + '.csv') == True:
			break

		headers = {'Referer' : 'http://marketdata.krx.co.kr/mdi',
					'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36'}

		gen_otp_url = "http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx"
		gen_otp_data = {'name' : "fileDown",
						'filetype' : "csv",
						'url' : "MKD/04/0404/04040200/mkd04040200_01",
						'market_gubun' : "STK",
						'indx_ind_cd' : "1001",
						'sect_tp_cd' : "ALL",
						'schdate' : date,
						'pagePath' : "/contents/MKD/04/0404/04040200/MKD04040200.jsp"}
		otp = requests.post(gen_otp_url, gen_otp_data, headers=headers)
		code = otp.content

		down_url = "http://file.krx.co.kr/download.jspx"
		down_data = {'code' : code,}
		down = requests.post(down_url, down_data, headers=headers)
		down.encoding = 'utf-8-sig'

		if len(down.content) > 1000:
			down = pd.read_csv(BytesIO(down.content), header=0, thousands=',')
			down = down.loc[:, ['종목코드', '종목명', '현재가','시가총액']]
			down.to_csv("CrawledData/" + date + '.csv', header=True, index=False)
			print(date)

	print("Update Crawling Finished")

if if __name__ == "__main__":
	update_crawl()