from enum import auto, StrEnum

import awswrangler as wr
from langchain.chains.llm import LLMChain
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI

from grip.common.config import config
from grip.common.exception import ValidationException
from grip.common.nlp.llm.generative_model import GenerativeModel
from grip.common.nlp.llm.message import Message, SystemMessage, HumanMessage, AIMessage
from grip.common.nlp.llm.callbacks import GripVertexAICallbackHandler, GripOpenAICallbackHandler


class OutputFormat(StrEnum):
    STRING = auto()
    JSON = auto()


class TextGenerator:
    """
    템플릿에 중괄호를 사용하면 input_data를 통해 치환이 가능하다.
    만약 프롬프트에 중괄호 자체를 사용하고 싶을때는 이중 중괄호({{|}})로 사용해야 한다.

    example usage:
    ```
    from textwrap import dedent

    text_generator = TextGenerator(
        model=Model.OPEN_AI_GPT_4,
        template_messages=[
            SystemMessage("너는 라이브커머스 플랫폼 업체의 데이터 분석가야."),
            HumanMessage(dedent(\"\"\"\
                다음은 그립이라는 라이브커머스 플랫폼에서 생성된 사용자 채팅 목록 샘플이야.
                이 채팅 목록을 상세 군집으로 구분하고 싶어.

                <messages>
                {chat_message_list}
                </messages>

                전체 메시지를 잘 살펴보고 어떤 군집들이 있는지 알려줘. 각 군집별 샘플 메시지를 5개씩 보여줘.
                필수적인 군집으로 제품 문의, 제품 칭찬, 제품 불만, 플랫폼 칭찬, 플랫폼 불만, 판매차 칭찬, 판매자 불만은 꼭 들어갔으면 좋겠어.
            \"\"\"))
        ],
    )

    sample_message_list = chat_df["message"].sample(2000).to_list()
    input_data = {"chat_message_list": sample_message_list}
    result = text_generator.generate_text(input_data=input_data)

    print(result)
    """

    def __init__(
        self,
        model: GenerativeModel,
        template_messages: list[Message],
        output_format=OutputFormat.STRING,
        verbose=True,
        max_tokens=2048,
        temperature=None
    ):
        self.model = model
        self.template_messages = template_messages
        self.output_format = output_format
        self.verbose = verbose
        self.temperature = None
        self._llm = None
        self._callbacks = None
        self._llm_chain = None

    def validate_template_messages(self) -> None:
        """
        템플릿 메시지 리스트의 유효성을 검사한다.
        1. 첫번째 메시지는 SystemMessage거나 UserMessage여야 한다.
        2. UserMessage 다음에는 AIMessage가 와야 한다.
        3. 마지막 메시지는 UserMessage여야 한다.
        """
        if not self.template_messages:
            raise ValidationException("Message list cannot be empty.")

        start_index = 1 if isinstance(self.template_messages[0], SystemMessage) else 0

        for i in range(start_index, len(self.template_messages), 2):
            if not isinstance(self.template_messages[i], HumanMessage): 
                raise ValidationException(
                    f"Expected a user message at position {i}, but found {self.template_messages[i]}."
                )

            if i + 1 < len(self.template_messages) and isinstance(self.template_messages[i + 1], AIMessage):
                raise ValidationException(
                    f"Expected an assistant message at position {i}, but found {self.template_messages[i + 1]}."
                )

        if not isinstance(self.template_messages[-1], HumanMessage):
            raise ValidationException("The last message must be a user message.")

    def get_llm(self) -> BaseChatModel:
        if self._llm:
            return self._llm

        match self.model:
            case GenerativeModel.OPEN_AI_GPT_3_5 | GenerativeModel.OPEN_AI_GPT_4 | GenerativeModel.OPEN_AI_GPT_4o | GenerativeModel.OPEN_AI_GPT_4o_MINI:
                # OpenAI API key는 고정된 key로 AWS Secrets Manager에 저장되어 있다.
                external_api_keys = wr.secretsmanager.get_secret_json("bigdata/external-api-keys")
                openai_api_key = external_api_keys["openai-api-key"]
                llm = ChatOpenAI(model_name=self.model, openai_api_key=openai_api_key, verbose=self.verbose, max_tokens=2048)
                
                if self.temperature:
                    llm.temperature = self.temperature

                if self.output_format == OutputFormat.JSON:
                    llm = llm.bind(response_format={"type": "json_object"})
            case GenerativeModel.VERTEX_GEMINI_1_0_PRO:
                # GCP의 경우 AWS Secrets Manager에 저장된 GCP service account key를 사용한다.
                # 이때 project id는 이미 credentials에 포함되어 있으나 VertexAI 객체 생성 시 반드시 필요하기때문에 추가해주어야 한다.
                credentials = config.gcp.get_credentials()

                llm = ChatVertexAI(
                    model_name=self.model,
                    credentials=credentials,
                    project=credentials.project_id,
                    location=config.gcp.default_region,
                    verbose=self.verbose,
                )

                if self.output_format == OutputFormat.JSON:
                    llm = llm.bind(response_mime_type="application/json")
            case _:
                raise ValueError("Unsupported model")

        self._llm = llm

        return llm

    def get_callbacks(self) -> list[BaseCallbackHandler]:
        if self._callbacks:
            return self._callbacks

        match self.model:
            case GenerativeModel.OPEN_AI_GPT_3_5 | GenerativeModel.OPEN_AI_GPT_4 | GenerativeModel.OPEN_AI_GPT_4o | GenerativeModel.OPEN_AI_GPT_4o_MINI:
                callbacks = [GripOpenAICallbackHandler()]
            case GenerativeModel.VERTEX_GEMINI_1_0_PRO:
                # VertexAI의 경우 LangChain 응답에 모델 정보가 없어서 콜백 클래스 생성 시 모델 정보를 넘겨주어야 한다.
                callbacks = [GripVertexAICallbackHandler(self.model)]
            case _:
                raise ValueError("Unsupported model")

        self._callbacks = callbacks

        return callbacks

    def get_llm_chain(self) -> LLMChain:
        if self._llm_chain:
            return self._llm_chain

        self.validate_template_messages()
        prompt_template = ChatPromptTemplate.from_messages(
            [(m.message_type, m.message) for m in self.template_messages]
        )
        llm = self.get_llm()

        match self.output_format:
            case OutputFormat.JSON:
                output_parser = JsonOutputParser()
            case OutputFormat.STRING:
                output_parser = StrOutputParser()
            case _:
                raise ValueError("Unsupported output format")

        llm_chain = LLMChain(prompt=prompt_template, llm=llm, output_parser=output_parser, verbose=self.verbose)
        self._llm_chain = llm_chain

        return llm_chain

    def generate_text(self, input_data) -> str | dict:
        conf = RunnableConfig(callbacks=self.get_callbacks())
        output_data = self.get_llm_chain().invoke(input_data, config=conf)

        return output_data["text"]

    async def generate_text_async(self, input_data) -> str | dict:
        conf = RunnableConfig(callbacks=self.get_callbacks())
        output_data = await self.get_llm_chain().ainvoke(input_data, config=conf)

        return output_data["text"]
