from abc import ABC

from pydantic import BaseModel, Field


class Message(BaseModel, ABC):
    message_type: str = Field(..., pattern="^(system|user|ai)$")
    message: str = ...

    def __init__(self, message: str, **data):
        super().__init__(message=message, **data)


class SystemMessage(Message):
    message_type: str = Field("system", frozen=True)


class UserMessage(Message):
    message_type: str = Field("user", frozen=True)


class AIMessage(Message):
    message_type: str = Field("ai", frozen=True)


# AssistantMessage와 HumanMessage로도 사용 가능
AssistantMessage = AIMessage
HumanMessage = UserMessage
