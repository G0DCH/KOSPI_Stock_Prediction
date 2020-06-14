# [local path]에 PredictPNG의 절대 경로를 대입
# [bucket path]에 aws s3의 버킷 주소를 대입
aws s3 sync [local path] s3://[bucket path] --acl public-read
