from datetime import datetime, timezone, timedelta

from grip.common.config import config

S3_DATA_BUCKET_NAME = config.aws.s3.data_bucket
S3_OUTPUT_BUCKET_NAME = config.aws.s3.output_bucket
TIMEZONE_KST = timezone(timedelta(hours=9), "Asia/Seoul")
TODAY_KST = datetime.now().astimezone(TIMEZONE_KST).date()
