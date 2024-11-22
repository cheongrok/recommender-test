from enum import StrEnum


class GenerativeModel(StrEnum):
    OPEN_AI_GPT_3_5 = "gpt-3.5-turbo"
    OPEN_AI_GPT_4 = "gpt-4-turbo"
    OPEN_AI_GPT_4o = "gpt-4o"
    OPEN_AI_GPT_4o_MINI = "gpt-4o-mini"
    VERTEX_GEMINI_1_0_PRO = "gemini-1.0-pro"
