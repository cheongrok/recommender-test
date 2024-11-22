from enum import Enum


class UserType(Enum):
    GUEST = 1
    CUSTOMER = 2
    GRIPPER = 3
    SELLER = 4
    ADMIN = 5
    CLOUD = 6
    CLOUD_GUEST = 7

    @classmethod
    def is_gripper_or_seller(cls, user_type: int):
        return (user_type == UserType.GRIPPER.value) | (user_type == UserType.SELLER.value)
