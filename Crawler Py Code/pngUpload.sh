# crontab에 등록해서 쓰는 경우
# PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# 이 문장을 crontab 첫줄에 대입하고 진행

export AWS_CONFIG_FILE="MY CONFIG FILE ROUTE" # 내 aws configure 파일의 절대 경로 대입
export AWS_ACCESS_KEY_ID=MY_ACCESS_KEY_ID # 내 credential.csv 파일에 적혀있는 액세스 키 및 시크릿 액세스키 대입
export AWS_SECRET_ACCESS_KEY=MY_SECRET_ACCESS_KEY_ID

# [local path]에 PredictPNG의 절대 경로를 대입
# [bucket path]에 aws s3의 버킷 주소를 대입
aws s3 sync [local path] s3://[bucket path] --acl public-read
