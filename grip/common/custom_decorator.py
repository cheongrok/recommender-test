import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] - %(message)s")
# INFO 로그가 너무 많이 남아서 일부 class의 loggin level 변경
logging.getLogger("botocore").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)


def elapsed_time(func):
    """
    function의 실행 시간을 출력해 주는 decorator
    :param func:
    :return:
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()

        function_code = func.__code__
        filename = Path(function_code.co_filename)
        logger = logging.getLogger(f"{filename.parent.name}/{filename.name}")
        logger.info(f"{func.__name__} 실행")
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} 실행 시간: {execution_time:0.2f} 초")

        return result

    return wrapper
