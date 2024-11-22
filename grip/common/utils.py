import argparse
from datetime import datetime, timedelta
from typing import Tuple, Union

from dateutil.rrule import rrule, DAILY

DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def date_list(start_date: Union[str, datetime.date], end_date: Union[str, datetime.date]):
    """start_date부터 end_date(미포함)까지의 date 객체 list를 리턴"""
    # rrule의 until은 끝 날짜를 포함하기 때문에 하루를 더 빼준다. 거의 대부분이 끝 날짜를 포함 안하기 때문에 관례적인 룰을 따름

    if type(start_date) is str:
        start_date = datetime.strptime(start_date, DATE_FORMAT).date()

    if type(end_date) is str:
        end_date = datetime.strptime(end_date, DATE_FORMAT).date()

    fixed_end_date = end_date - timedelta(days=1)

    return map(lambda x: x.date(), rrule(DAILY, dtstart=start_date, until=fixed_end_date))


def chunk_list(lst, chunk_size):
    """
    리스트를 chunk_size 만큼 나누어서 각 단위리스트를 순서대로 yield
    :param lst:
    :param chunk_size:
    :return:
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i: i + chunk_size]


def parse_args_date_1(
        key: str = "start_date",
        key_short: str = None,
        description: str = None,
) -> datetime.date:
    parser = argparse.ArgumentParser(description=description)
    if key_short is None:
        parser.add_argument(f"--{key}", required=True)
    else:
        parser.add_argument(f"-{key_short}", f"--{key}", required=True)

    args = parser.parse_args()
    key_date = datetime.strptime(vars(args)[f"{key}"], DATE_FORMAT).date()

    return key_date


def parse_args_date_2(
        key1: str = "start_date",
        key1_short: str = None,
        key2: str = "end_date",
        key2_short: str = None,
        description: str = None,
) -> tuple[datetime.date, datetime.date]:
    parser = argparse.ArgumentParser(description=description)
    if key1_short is None:
        parser.add_argument(f"--{key1}")
    else:
        parser.add_argument(f"-{key1_short}", f"--{key1}")

    if key2_short is None:
        parser.add_argument(f"--{key2}")
    else:
        parser.add_argument(f"-{key2_short}", f"--{key2}")

    args = parser.parse_args()
    key1_date = datetime.strptime(vars(args)[f"{key1}"], DATE_FORMAT).date()
    key2_date = datetime.strptime(vars(args)[f"{key2}"], DATE_FORMAT).date()

    return key1_date, key2_date


def parse_args_datetime_2(
        key1: str = "start_date",
        key1_short: str = None,
        key2: str = "end_date",
        key2_short: str = None,
        description: str = None,
) -> Tuple[datetime, datetime]:
    parser = argparse.ArgumentParser(description=description)
    if key1_short is None:
        parser.add_argument(f"--{key1}")
    else:
        parser.add_argument(f"-{key1_short}", f"--{key1}")

    if key2_short is None:
        parser.add_argument(f"--{key2}")
    else:
        parser.add_argument(f"-{key2_short}", f"--{key2}")

    args = parser.parse_args()
    key1_date = datetime.strptime(vars(args)[f"{key1}"], DATETIME_FORMAT)
    key2_date = datetime.strptime(vars(args)[f"{key2}"], DATETIME_FORMAT)

    return key1_date, key2_date


def parse_args_date_4(
        key1: str, key2: str, key3: str, key4: str, description: str
) -> Tuple[datetime.date, datetime.date, datetime.date, datetime.date]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(f"--{key1}")
    parser.add_argument(f"--{key2}")
    parser.add_argument(f"--{key3}")
    parser.add_argument(f"--{key4}")
    args = parser.parse_args()

    key1_date = datetime.strptime(vars(args)[f"{key1}"], DATE_FORMAT).date()
    key2_date = datetime.strptime(vars(args)[f"{key2}"], DATE_FORMAT).date()
    key3_date = datetime.strptime(vars(args)[f"{key3}"], DATE_FORMAT).date()
    key4_date = datetime.strptime(vars(args)[f"{key4}"], DATE_FORMAT).date()

    return key1_date, key2_date, key3_date, key4_date
