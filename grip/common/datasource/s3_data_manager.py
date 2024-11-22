import glob
import inspect
import logging
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Union, Any

import awswrangler as wr
import pandas as pd

from grip.common.config import config
from grip.common.datasource.grip_db_table import GripDbTable

SEARCH_QUERY_PATH = "search/query"


class S3DataManager:
    @classmethod
    def date_path(cls, date: datetime.date):
        return f"year={date.year}/month={date.month}/day={date.day}"

    @classmethod
    def month_path(cls, date: datetime.date):
        return f"year={date.year}/month={date.month}"

    @classmethod
    def s3_key_by_day(cls, bucket: str, path: str, date: datetime.date, filename: str = ""):
        return f"s3://{bucket}/{path}/{cls.date_path(date)}/{filename}"

    @classmethod
    def s3_key_by_month(cls, bucket: str, path: str, date: datetime.date):
        return f"s3://{bucket}/{path}/year={date.year}/month={date.month}"

    @classmethod
    def s3_key_grip_db_latest(cls, table: GripDbTable):
        return f"s3://{config.aws.s3.data_bucket}/grip_db/latest/{table.value}/"

    @classmethod
    def s3_key_grip_db_by_date(cls, table: GripDbTable, date: datetime.date):
        return f"s3://{config.aws.s3.data_bucket}/grip_db/{table.value}/{cls.date_path(date)}/"

    @classmethod
    def s3_key_search_query(cls, date: datetime.date):
        """S3 검색 쿼리 히스토리 key"""
        return cls.s3_key_by_day(config.aws.s3.output_bucket, SEARCH_QUERY_PATH, date)

    @classmethod
    def s3_read_view_content(cls, date: datetime.date, **kwargs) -> pd.DataFrame:
        prefix = f"elasticSearch/viewContent/{cls.date_path(date)}"
        return cls.s3_read_parquet(f"s3://{config.aws.s3.output_bucket}/{prefix}/", **kwargs)

    @classmethod
    def s3_read_view_product(cls, date: datetime.date, **kwargs) -> pd.DataFrame:
        prefix = f"RecSys_product/VIEW_PRODUCT/{cls.date_path(date)}"
        return cls.s3_read_parquet(f"s3://{config.aws.s3.output_bucket}/{prefix}/", **kwargs)

    @classmethod
    def s3_read_order_all(cls, date: datetime.date) -> pd.DataFrame:
        prefix = f"order/all/{cls.date_path(date)}"
        return wr.s3.read_csv(f"s3://{config.aws.s3.data_bucket}/{prefix}/")

    @classmethod
    def s3_read_order_all_month(cls, date: datetime.date) -> pd.DataFrame:
        prefix = f"order/all/{cls.month_path(date)}"
        return wr.s3.read_csv(f"s3://{config.aws.s3.data_bucket}/{prefix}/")

    # ------------------------------ DOWNLOAD ------------------------------

    @classmethod
    def s3_read_parquet(
        cls,
        s3_key: str,
        partition_date: datetime.date = None,
        safe=False,
        **kwargs,
    ) -> pd.DataFrame:
        if partition_date is not None:
            s3_key = s3_key[:-1] if s3_key[-1] == "/" else s3_key
            s3_key = f"{s3_key}/{cls.date_path(partition_date)}"

        # safe 파라미터는 aws wrangler의 버전에 따라 다르게 적용을 해야 해서 분기처리
        if "safe" in inspect.signature(wr.s3.read_parquet).parameters:
            return wr.s3.read_parquet(s3_key, safe=safe, **kwargs)

        return wr.s3.read_parquet(s3_key, pyarrow_additional_kwargs={"safe": safe}, **kwargs)

    @classmethod
    def s3_read_csv(
        cls,
        s3_key: str,
        partition_date: datetime.date = None,
        **kwargs,
    ) -> pd.DataFrame:
        if partition_date is not None:
            s3_key = s3_key[:-1] if s3_key[-1] == "/" else s3_key

        s3_key = f"{s3_key}/{cls.date_path(partition_date)}"

        return wr.s3.read_csv(s3_key, **kwargs)

    @classmethod
    def s3_check_path(cls, s3_key: str, partition_date: datetime.date, file_name: str) -> bool:
        if partition_date is not None:
            s3_key = s3_key[:-1] if s3_key[-1] == "/" else s3_key
            s3_key = f"{s3_key}/{cls.date_path(partition_date)}/{file_name}"
        return wr.s3.does_object_exist(s3_key)

    @classmethod
    def s3_read_search_query(cls, date: datetime.date, **kwargs) -> pd.DataFrame:
        """특정 날짜의 검색 쿼리 히스토리를 다운로드"""
        return cls.s3_read_parquet(cls.s3_key_search_query(date), **kwargs)

    @classmethod
    def s3_read_grip_db_table_by_date(cls, table: GripDbTable, date: datetime.date, **kwargs) -> pd.DataFrame:
        return cls.s3_read_parquet(cls.s3_key_grip_db_by_date(table, date), **kwargs)

    @classmethod
    def s3_read_grip_db_latest_table(cls, table: GripDbTable, **kwargs) -> pd.DataFrame:
        return cls.s3_read_parquet(cls.s3_key_grip_db_latest(table), **kwargs)

    @classmethod
    def s3_download(cls, path: str, local_file: Union[str, Any]):
        """
        S3의 특정 파일 다운로드

        Examples
        --------
        >>> S3DataManager.s3_download(path='s3://bucket/key', local_file='./key')
        ---- OR ----
        >>> with open(file='./key', mode='wb') as local_f:
        >>>     S3DataManager.s3_download(path='s3://bucket/key', local_file=local_f)
        """

        wr.s3.download(path, local_file)

    # ------------------------------ UPLOAD ------------------------------

    @classmethod
    def s3_upload(cls, file: str, s3_key: str):
        """S3에 file 업로드"""
        wr.s3.upload(file, s3_key)

        logging.info(f"{file} uploaded to {s3_key}")

    @classmethod
    def s3_upload_parquet(cls, df: pd.DataFrame, s3_key: str):
        """S3에 parquet 업로드"""
        wr.s3.to_parquet(
            df=df,
            path=s3_key,
        )

    @classmethod
    def s3_upload_partitioned_by_date(
        cls, file: str, bucket: str, prefix: str, file_name: str, date: datetime.date
    ) -> str:
        """
        날짜에 해당하는 디렉토리에 파일을 업로드, 하둡 하이브에서 사용하는 파티셔닝 컨벤션을 따름
         - s3://{bucket}/{prefix}/year={}/month={}/day={}/{file_name}
        """
        s3_key = f"s3://{bucket}/{prefix}/{cls.date_path(date)}/{file_name}"
        cls.s3_upload(file, s3_key)

        return s3_key

    @classmethod
    def s3_zip_and_upload_model(cls, files: str, bucket: str, prefix: str, file_name: str, date: datetime.date):
        with tarfile.open(file_name, "w:gz") as tar:
            for file in glob.glob(files + "*"):
                tar.add(file)

        wr.s3.upload(
            file_name,
            f"s3://{bucket}/{prefix}/{cls.date_path(date)}/{file_name}",
        )

    @classmethod
    def s3_upload_directory(
        cls, directory: str, bucket: str, prefix: str, date: datetime.date = None, target_dir: str = None
    ) -> str:
        """
        s3://{bucket}/{prefix}/{target_dir} 에 directory에 있는 모든 파일을 업로드
        date가 지정되면 날짜에 해당하는 디렉토리에 파일을 업로드, 하둡 하이브에서 사용하는 파티셔닝 컨벤션을 따름
        s3://{bucket}/{prefix}/year={}/month={}/day={}/{target_dir}
        """
        if not directory.endswith("/"):
            directory += "/"

        path = Path(directory)
        for file in path.glob("**/*"):
            if file.is_file():
                if date is None:
                    if target_dir:
                        s3_key = f"s3://{bucket}/{prefix}/{target_dir}/{file.relative_to(path)}"
                    else:
                        s3_key = f"s3://{bucket}/{prefix}/{file.relative_to(path)}"
                else:
                    if target_dir:
                        s3_key = f"s3://{bucket}/{prefix}/{cls.date_path(date)}/{target_dir}/{file.relative_to(path)}"
                    else:
                        s3_key = f"s3://{bucket}/{prefix}/{cls.date_path(date)}/{file.relative_to(path)}"

                cls.s3_upload(str(file), s3_key)

        return f"s3://{bucket}/{prefix}/"

    # ------------------------------ ETC ------------------------------
    @classmethod
    def s3_copy_an_object(cls, source_key: str, target_path: str):
        """
        source_key에 명시된 하나의 파일을 target_path에 지정된 디렉토리로 copy한다.
        파일 이름은 변경할 수 없다.
        :param source_key: s3://bucket/path/to/file
        :param target_path: s3://bucket/path *파일 이름은 포함하지 않는다*
        """
        wr.s3.copy_objects(
            paths=[source_key],
            source_path=source_key[: source_key.rfind("/")],
            target_path=target_path,
        )
