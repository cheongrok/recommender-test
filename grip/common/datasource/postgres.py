import logging
import os
from enum import StrEnum
from urllib.parse import quote_plus

import awswrangler as wr
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.dialects.postgresql import insert


class Postgres:
    class Instance(StrEnum):
        PROD = "aibigdata-prod"
        ANALYTICS = "aibigdata-analytics"

    @classmethod
    def get_engine(cls, *, instance=Instance.PROD, database: str):
        if os.environ.get("PHASE") == "local":
            match instance:
                case cls.Instance.PROD:
                    port = 40050
                case cls.Instance.ANALYTICS:
                    port = 40051
                case _:
                    raise ValueError(f"Unknown instance: {instance}")

            uri = f"postgresql+psycopg2://querypie:querypie@localhost:{port}"
        else:
            secret = wr.secretsmanager.get_secret_json(f"bigdata/{instance}")
            uri = f"postgresql+psycopg2://{secret['username']}:{quote_plus(secret['password'])}@{secret['host']}:{secret['port']}"

        if database is not None:
            uri = f"{uri}/{database}"

        engine = create_engine(uri)

        return engine

    @classmethod
    def read_sql(cls, sql: str, *, instance=Instance.PROD, database="common") -> pd.DataFrame:
        logging.debug(sql)

        with cls.get_engine(instance=instance, database=database).connect() as conn:
            result_df = pd.read_sql(sql, conn)

        return result_df

    @classmethod
    def insert_data(
        cls,
        df,
        instance,
        database,
        table_name,
        on_conflict_do_update_index: list[str] | None = None,
        on_conflict_do_update_set_keys: list[str] | None = None,
    ):
        engine = cls.get_engine(instance=instance, database=database)
        metadata = MetaData()
        table = Table(table_name, metadata, autoload_with=engine)
        with engine.connect() as connection:
            stmt = insert(table).values(df.to_dict(orient="records"))

            if on_conflict_do_update_index is not None and on_conflict_do_update_set_keys is not None:
                stmt = stmt.on_conflict_do_update(
                    index_elements=on_conflict_do_update_index,
                    set_={key: stmt.excluded[key] for key in on_conflict_do_update_set_keys},
                )

            connection.execute(stmt)
            connection.commit()
