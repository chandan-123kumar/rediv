"""FastAPI application for the local support assistant."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from application.pipeline import SupportService, create_service
from application.schemas import AnswerRequest, AnswerResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.support_service = create_service()
    try:
        yield
    finally:
        app.state.support_service.close()


app = FastAPI(title="Local AI Support Assistant", lifespan=lifespan)


@app.post("/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest, http_request: Request):
    service: SupportService = http_request.app.state.support_service
    try:
        return service.answer(request.question)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Support assistant is temporarily unavailable") from exc
