import logging
from textwrap import dedent
from typing import Any

from langchain_core.outputs import LLMResult
from langchain_google_vertexai.callbacks import VertexAICallbackHandler
from langchain_community.callbacks.openai_info import OpenAICallbackHandler

from grip.common.nlp.llm.generative_model import GenerativeModel

MODEL_INPUT_COST_PER_1K_TOKENS_USD = {
    GenerativeModel.VERTEX_GEMINI_1_0_PRO: 0.000125,
}

MODEL_OUTPUT_COST_PER_1K_TOKENS_USD = {
    GenerativeModel.VERTEX_GEMINI_1_0_PRO: 0.000375,
}


class GripVertexAICallbackHandler(VertexAICallbackHandler):
    """
    기본 콜백 클래스는 비용 정보를 출력하지 않기 때문에 비용 정보를 출력할 수 있도록 기존 클래스를 상속받아 구현한 클래스
    추가로 기존 클래스와 달리 수동으로 출력을 추가해줘야 할 필요 없이 기본적으로 INFO 레벨에 출력하도록 수정
    """

    def __init__(self, generative_model: GenerativeModel, usd_won_exchange_rate=1350.0) -> None:
        super().__init__()
        self.model = generative_model
        self.total_cost = 0.0
        self.usd_won_exchange_rate = usd_won_exchange_rate

    def __repr__(self) -> str:
        super_repr = super().__repr__()
        krw_total_cost = self.total_cost * self.usd_won_exchange_rate

        return (
            "\n"
            f"{super_repr}"
            f"Total Cost (USD): ${self.total_cost:.6f}\n"
            f"Total Cost (KRW): {krw_total_cost:.3f}원 (환율 {int(self.usd_won_exchange_rate)}원 기준)\n"
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        super().on_llm_end(response, **kwargs)

        if self.model in MODEL_INPUT_COST_PER_1K_TOKENS_USD and self.model in MODEL_OUTPUT_COST_PER_1K_TOKENS_USD:
            prompt_cost = self.prompt_tokens / 1000 * MODEL_INPUT_COST_PER_1K_TOKENS_USD[self.model]
            completion_cost = self.completion_tokens / 1000 * MODEL_OUTPUT_COST_PER_1K_TOKENS_USD[self.model]
            total_cost = prompt_cost + completion_cost
        else:
            total_cost = 0.0

        with self._lock:
            self.total_cost = total_cost

        logging.info(self)


class GripOpenAICallbackHandler(OpenAICallbackHandler):
    """
    기본 콜백 클래스는 비용 정보를 출력하지만 달러로만 출력하기 때문에 원단위 출력을 추가하도록 기존 클래스를 상속받아 구현한 클래스
    추가로 기존 클래스와 달리 수동으로 출력을 추가해줘야 할 필요 없이 기본적으로 INFO 레벨에 출력하도록 수정
    """

    def __init__(self, usd_won_exchange_rate=1350.0) -> None:
        super().__init__()
        self.usd_won_exchange_rate = usd_won_exchange_rate

    def __repr__(self) -> str:
        super_repr = super().__repr__()
        krw_total_cost = self.total_cost * self.usd_won_exchange_rate

        return (
            "\n"
            f"{super_repr}\n"
            f"Total Cost (KRW): {krw_total_cost:.3f}원 (환율 {int(self.usd_won_exchange_rate)}원 기준)\n"
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        super().on_llm_end(response, **kwargs)

        logging.info(self)
