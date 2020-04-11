library(httr)
library(rvest)
library(readr)
library(readxl)

	date = format(Sys.Date(), "%Y%m%d")
	gen_otp_url = "http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx"
	gen_otp_data = list(name = "fileDown",
	                    filetype = "csv",
                            url = "MKD/04/0404/04040200/mkd04040200_01",
                            market_gubun = "STK",
                            indx_ind_cd = "1028",
                            sect_tp_cd = "ALL",
                            schdate = date,
                            pagePath = "/contents/MKD/04/0404/04040200/MKD04040200.jsp")
	otp = POST(gen_otp_url, query = gen_otp_data) %>%
		          read_html() %>% html_text()

	down_url = "http://file.krx.co.kr/download.jspx"
        down_data = list(
                               code = otp
                        )
        down = POST(down_url, query = down_data,
                    add_headers(referer = gen_otp_url)) %>% 
		    read_html() %>% html_text() 
	    #%>% read_csv()

	if(nchar(down) > 1000)
	{
		down = read_csv(down)
		data_sector = down[, c('종목코드', '종목명', '현재가','시가총액')]
		write.csv(data_sector, paste("/home/ubuntu/crawled_data/crawled_data/", date, ".csv", sep = ""))
		print(date)
	}
