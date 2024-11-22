from enum import Enum

import awswrangler as wr
from hashids import Hashids


class CryptorType(str, Enum):
    RESERVATION = "reservation_id"
    CONTENT = "content_id"
    USER = "user_id"
    GRIPRUN = "griprun_id"
    PRODUCT_ID = "product_id"


class CommonIdCryptor:
    def __init__(self, cryptor: CryptorType):
        hash_param = wr.secretsmanager.get_secret_json("common/hashids")[cryptor]
        self.hash_ids = Hashids(
            salt=hash_param["salt"],
            min_length=hash_param["min_length"],
            alphabet=hash_param["alphabet"],
        )

    def encrypt(self, common_seq):
        return self.hash_ids.encode(common_seq)

    def decrypt(self, common_id):
        if not common_id:
            return None

        decoded = self.hash_ids.decode(common_id)
        if not decoded:
            return None
        return decoded[0]
