import os
import tomllib
from dataclasses import dataclass
from enum import StrEnum, auto


class Phase(StrEnum):
    LOCAL_PROD = auto()
    LOCAL_DEV = auto()
    DEV = auto()
    TEST = auto()
    PROD = auto()

    def is_local_prod(self) -> bool:
        return self == Phase.LOCAL_PROD

    def is_local_dev(self) -> bool:
        return self == Phase.LOCAL_DEV

    def is_local(self) -> bool:
        return self.is_local_prod() or self.is_local_dev()

    def is_dev(self) -> bool:
        return self == Phase.DEV

    def is_test(self) -> bool:
        return self == Phase.TEST

    def is_prod(self) -> bool:
        return self == Phase.PROD

    def is_profile_credentials(self) -> bool:
        return self.is_local() or self.is_test()


@dataclass
class AwsConfig:
    credentials_provider: str
    default_region: str
    secrets_manager: dict
    s3: dict


@dataclass
class AwsSecretsManagerName:
    clickhouse: str
    gcp_credentials: str


@dataclass
class AwsS3:
    data_bucket: str
    output_bucket: str
    log_bucket: str
    legacy_data_bucket: str


@dataclass
class AwsConfig:
    credentials_provider: str
    default_region: str
    secrets_manager_name: AwsSecretsManagerName
    s3: AwsS3


@dataclass
class GcpConfig:
    project_id: str
    analytics_dataset_id: str
    default_region: str

    @classmethod
    def get_credentials_dict(cls) -> dict:
        import awswrangler as wr

        return wr.secretsmanager.get_secret_json(config.aws.secrets_manager_name.gcp_credentials)

    @classmethod
    def get_credentials_base64(cls) -> str:
        import base64
        import json

        credentials_dict = cls.get_credentials_dict()
        credentials_json = json.dumps(credentials_dict)
        credentials_bytes = credentials_json.encode("utf-8")
        credentials_base64_bytes = base64.b64encode(credentials_bytes)

        # base64 bytes to string
        return credentials_base64_bytes.decode("utf-8")

    @classmethod
    def get_credentials(cls):
        from google.oauth2.service_account import Credentials

        # from google.auth.credentials import Credentials

        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        credentials_dict = cls.get_credentials_dict()
        credentials = Credentials.from_service_account_info(credentials_dict, scopes=scopes)

        return credentials


@dataclass
class Config:
    phase: Phase
    aws: AwsConfig
    gcp: GcpConfig


def load_config(phase: Phase) -> Config:
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(f"{current_file_directory}/config/config-{phase.value}.toml")
    with open(file_path, "rb") as f:
        config_data = tomllib.load(f)

    if phase.is_profile_credentials():
        os.environ["AWS_PROFILE"] = config_data["aws"]["aws_profile"]

    aws_config = AwsConfig(
        credentials_provider=config_data["aws"]["aws_credentials_provider"],
        default_region=config_data["aws"]["default_region"],
        secrets_manager_name=AwsSecretsManagerName(
            clickhouse=config_data["aws"]["secrets_manager_name"]["clickhouse"],
            gcp_credentials=config_data["aws"]["secrets_manager_name"]["gcp_credential"],
        ),
        s3=AwsS3(
            data_bucket=config_data["aws"]["s3"]["data_bucket"],
            output_bucket=config_data["aws"]["s3"]["output_bucket"],
            log_bucket=config_data["aws"]["s3"]["log_bucket"],
            legacy_data_bucket=config_data["aws"]["s3"]["legacy_data_bucket"],
        ),
    )

    gcp_config = GcpConfig(
        project_id=config_data["gcp"]["project_id"],
        analytics_dataset_id=config_data["gcp"]["analytics_dataset_id"],
        default_region=config_data["gcp"]["default_region"],
    )

    return Config(phase=phase, aws=aws_config, gcp=gcp_config)


# Load the configuration
# unit test 환경에서도 파일을 잘 찾을 수 있도록 __file__과 상대경로를 이용
config = load_config(phase=Phase(os.getenv("PHASE", "local_dev")))
