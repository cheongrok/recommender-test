from datetime import timedelta, datetime
from enum import Enum

import awswrangler as wr
import pandas as pd

from grip.common.env import TIMEZONE_KST


class DynamoDbTable(str, Enum):
    RECOMMEND_CATEGORY = "recommend.category"
    CONTENT_BASED_RECOMMEND_RELATED_PRODUCT = "recommend.cb_related_products"
    RECOMMEND_RELATED_GRIPPER = "recommend.related_gripper"
    RECOMMEND_RELATED_PRODUCT = "recommend.related_products"
    RECOMMEND_TOP_SCORE_GRIPPER = "recommend.user_top_score_gripper"
    SEARCH_RELATED_GRIPPER = "search.related_gripper"
    # RECOMMEND_USER_VIEW_HISTORY = "recommend.user_view_history" # 미사용


class DynamoDbDataLoader:
    @classmethod
    def upload_df(cls, df: pd.DataFrame, table: DynamoDbTable, expired_in_days: int = None):
        if expired_in_days is not None:
            expired_at = (datetime.now().astimezone(TIMEZONE_KST) + timedelta(days=expired_in_days)).timestamp()
            df["expired_at"] = int(expired_at)

        wr.dynamodb.put_df(df, table.value)
