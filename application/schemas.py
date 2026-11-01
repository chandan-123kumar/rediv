"""FastAPI request and response schemas."""

from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1)


class AnswerResponse(BaseModel):
    question_id: str
    answer: str
    citations: list[str]
    insufficient_context: bool
