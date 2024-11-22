from enum import Enum

import awswrangler as wr
from langchain_openai import OpenAIEmbeddings


class EmbeddingModel(str, Enum):
    OPEN_AI_EMBEDDING_3_SMALL = "text-embedding-3-small"
    OPEN_AI_EMBEDDING_3_LARGE = "text-embedding-3-large"
    OPEN_AI_EMBEDDING_ADA_2 = "text-embedding-ada-002"


class EmbeddingGenerator:
    def __init__(self, model: EmbeddingModel, dimensions=None):
        self.model = model

        match model:
            case (
                EmbeddingModel.OPEN_AI_EMBEDDING_3_SMALL
                | EmbeddingModel.OPEN_AI_EMBEDDING_3_LARGE
                | EmbeddingModel.OPEN_AI_EMBEDDING_ADA_2
            ):
                external_api_keys = wr.secretsmanager.get_secret_json("bigdata/external-api-keys")
                openai_api_key = external_api_keys["openai-api-key"]

                kwargs = {"model": model.value, "openai_api_key": openai_api_key}

                if dimensions is not None:
                    kwargs.update({"dimensions": dimensions})

                self.embedding_model = OpenAIEmbeddings(**kwargs)
            case _:
                raise ValueError("Unsupported model")

    def embed_documents(self, texts: list[str]):
        result = self.embedding_model.embed_documents(texts)

        return result
