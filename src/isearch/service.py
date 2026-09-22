"""FastAPI service: ask / search / documents / feedback / admin reports."""
from __future__ import annotations

import uuid
from typing import Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .agent import Assistant
from .config import Settings

Role = Literal["public", "student", "staff"]


class Turn(BaseModel):
    q: str
    a: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    role: Role = "student"
    category: str | None = None
    history: list[Turn] = Field(default_factory=list, max_length=20)
    use_llm: bool | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    role: Role = "student"
    category: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class FeedbackRequest(BaseModel):
    query_id: int
    rating: Literal[1, -1]
    comment: str = ""


def build_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    assistant = Assistant(settings)
    app = FastAPI(title="Institutional AI Search", version="0.1.0")
    app.state.assistant = assistant

    def require_admin(x_api_key: str | None = Header(default=None)) -> None:
        if settings.admin_key and x_api_key != settings.admin_key:
            raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")

    @app.get("/health")
    def health():
        return {"status": "ok", "documents": len(assistant.docs), "llm_available": assistant.llm.available()}

    @app.post("/ask")
    def ask(req: AskRequest):
        r = assistant.ask(req.question, history=[t.model_dump() for t in req.history],
                          session_id=req.session_id or str(uuid.uuid4()), role=req.role,
                          category=req.category, use_llm=req.use_llm)
        return r.to_dict()

    @app.post("/search")
    def search(req: SearchRequest):
        return {"results": assistant.search(req.query, category=req.category, role=req.role, top_k=req.top_k)}

    @app.post("/feedback")
    def feedback(req: FeedbackRequest):
        try:
            assistant.store.add_feedback(req.query_id, req.rating, req.comment)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"status": "recorded"}

    @app.get("/documents")
    def documents():
        return {"documents": assistant.documents()}

    @app.post("/documents", dependencies=[Depends(require_admin)])
    async def upload_document(category: str = "general", file: UploadFile = File(...)):
        try:
            info = assistant.save_document(file.filename, await file.read(), category)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return info

    @app.delete("/documents/{path:path}", dependencies=[Depends(require_admin)])
    def delete_document(path: str):
        try:
            assistant.remove_document(path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"status": "removed"}

    @app.post("/reload", dependencies=[Depends(require_admin)])
    def reload_index():
        assistant.reload()
        return {"documents": len(assistant.docs)}

    @app.get("/admin/stats", dependencies=[Depends(require_admin)])
    def stats():
        return assistant.store.stats()

    @app.get("/admin/unanswered", dependencies=[Depends(require_admin)])
    def unanswered(limit: int = 50):
        return {"unanswered": assistant.store.unanswered(limit)}

    @app.get("/admin/downvoted", dependencies=[Depends(require_admin)])
    def downvoted(limit: int = 50):
        return {"downvoted": assistant.store.downvoted(limit)}

    return app
