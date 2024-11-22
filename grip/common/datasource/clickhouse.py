from enum import StrEnum, auto
from datetime import datetime
from typing import List

import awswrangler as wr
import pandas as pd
from sqlalchemy import create_engine

from grip.common.config import config
from urllib.parse import quote_plus

from grip.common.datasource.grip_db_table import GripDbTable
from grip.common.datasource.griprun_db_table import GripRunDbTable


class ClickhouseSchema(StrEnum):
    default = auto()
    grip_db = auto()
    grip_db_realtime = auto()
    grip_run_db = auto()
    ai_bigdata = auto()
    data_anal = auto()
    finance = auto()
    marketing = auto()
    md = auto()
    dimension = auto()


class DefaultTable(StrEnum):
    events_all = auto()
    order_all = auto()
    elasticsearch = auto()
    elasticsearch_seller = auto()
    producer = auto()


class Clickhouse:
    schema = ClickhouseSchema
    default_table = DefaultTable
    grip_db_table = GripDbTable
    grip_run_db_table = GripRunDbTable

    @classmethod
    def get_secret(cls) -> dict:
        return wr.secretsmanager.get_secret_json(config.aws.secrets_manager_name.clickhouse)

    @classmethod
    def get_engine(cls):
        secret = wr.secretsmanager.get_secret_json(config.aws.secrets_manager_name.clickhouse)
        uri = f"clickhouse+native://{secret['username']}:{quote_plus(secret['password'])}@{secret['host']}:{secret['port']}"
        engine = create_engine(uri)

        return engine

    @classmethod
    def run_sql(cls, sql: str) -> pd.DataFrame:
        print(f"===== running sql in Clickhouse=====\n{sql}")

        with cls.get_engine().connect() as conn:
            result_df = pd.read_sql(sql, conn)

        print(f"=====result of sql=====\n{result_df.head(10)}")

        return result_df

    @classmethod
    def get_contents(cls, publish_date: datetime.date, columns: List[str] = None):
        if columns is None:
            columns = [
                "content_seq",
                "content_id",
                "user_seq",
                "title",
                "created_at",
                "published_at",
            ]

        sql = f"""
            SELECT {', '.join(columns)}
            FROM grip_db_realtime.content
            WHERE date(published_at) = '{publish_date}'
        """

        content_df = cls.run_sql(sql)

        return content_df

    @classmethod
    def get_gmv_by_product(cls, date: datetime.date, include_flash_product=False) -> pd.DataFrame:
        sql = f"""
            WITH order_list AS (
                SELECT product_seq, sum(gmv + shipping) AS gmv
                FROM default.order_all
                WHERE date(ordered_at) = '{date}'
                GROUP BY product_seq
            )
            SELECT ol.product_seq AS product_seq
                , pi.user_seq AS user_seq
                , user_name
                , pi.product_name AS product_name
                , gmv
            FROM order_list ol
                LEFT JOIN grip_db.product_info pi ON ol.product_seq = pi.product_seq
                LEFT JOIN grip_db.member m ON pi.user_seq = m.user_seq
            WHERE TRUE \
                {"" if include_flash_product else "AND pi.flash = 'N'"}
            ORDER BY gmv DESC
        """

        return cls.run_sql(sql)

    @classmethod
    def get_modified_product(cls, date: datetime.date, include_flash_product=False) -> pd.DataFrame:
        sql = f"""
            SELECT pi.product_seq AS product_seq
                , pi.user_seq AS user_seq
                , user_name
                , pi.product_name AS product_name
            FROM grip_db.product_info pi LEFT JOIN grip_db.member m ON pi.user_seq = m.user_seq
            WHERE date(pi.modified_at) = '{date}'
                {"" if include_flash_product else "AND pi.flash = 'N'"}
            ORDER BY pi.modified_at DESC
        """

        return cls.run_sql(sql)
