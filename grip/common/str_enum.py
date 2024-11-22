from enum import Enum


class AutoLowerStrEnum(str, Enum):
    # noinspection PyMethodParameters
    # 첫번째 파라미터가 self가 아니라 뜨는 경고 제거
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()
