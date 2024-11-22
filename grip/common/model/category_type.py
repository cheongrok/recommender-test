from enum import Enum


class CategoryType(Enum):
    """
    category 테이블의 category_type 칼럼에서 사용되는 값
    """

    PRODUCT = 1
    CONTENT = 2
