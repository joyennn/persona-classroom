from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Optional
from datetime import datetime
import os
import uvicorn
import traceback

# ─────────────────────────────────────────────────────────────
# OpenAI Python SDK v1.x 사용 (pip install -U openai)
# ─────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None
    print("[BOOT] OpenAI SDK import 실패:", e)

# ─────────────────────────────────────────────────────────────
# 멀티 페르소나(교사 + 학생들) 정의 — 개발자가 고정 설정
#   - image_url은 /static/* 에 있는 파일을 가리키도록 준비
# ─────────────────────────────────────────────────────────────
PERSONAS: List[Dict] = [
    {
        "id": "teacher",
        "name": "선생님",
        "role": "teacher",
        "image_url": "/static/teacher.jpg",
        "short_desc": "담화를 이끌고 학생 발화를 촉진하는 교사.",
        "system_prompt": (
            "너는 교사다. 학생들이 이해하도록 요점을 정리하고 질문하며, "
            "과도한 간섭 없이 학생 발화를 유도한다. 항상 한국어로 답한다."
        ),
    },
    {
        "id": "s1",
        "name": "학생1",
        "role": "student",
        "image_url": "/static/student1.jpg",
        "short_desc": "신중하고 정리된 말투. 발화는 짧고 핵심 위주.",
        "system_prompt": (
            "너는 학생1이다. 차분하고 신중한 말투로 핵심만 짧게 말한다. 항상 한국어로 답한다."
        ),
    },
    {
        "id": "s2",
        "name": "학생2",
        "role": "student",
        "image_url": "/static/student2.jpg",
        "short_desc": "호기심이 많고 질문을 자주 한다.",
        "system_prompt": (
            "너는 학생2이다. 호기심이 많아 이해 안 되는 점을 바로 묻는다. 항상 한국어로 답한다."
        ),
    },
    {
        "id": "s3",
        "name": "학생3",
        "role": "student",
        "image_url": "/static/student3.jpg",
        "short_desc": "말수가 적고 꼭 필요할 때만 한마디 한다.",
        "system_prompt": (
            "너는 학생3이다. 필요할 때만 짧게 말한다. 군더더기 없이 한국어로 답한다."
        ),
    },
]

# ─────────────────────────────────────────────────────────────
# FastAPI 앱 기본 셋업
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="Persona Classroom Chat", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일(이미지) 제공
app.mount("/static", StaticFiles(directory="static"), name="static")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(BASE_DIR, "index.html")


# ─────────────────────────────────────────────────────────────
# 요청/응답 모델
# ─────────────────────────────────────────────────────────────
Role = Literal["system", "user", "assistant"]

class Message(BaseModel):
    role: Role
    content: str

class VerifyRequest(BaseModel):
    api_key: str = Field(min_length=10)

class ChatRequest(BaseModel):
    api_key: str = Field(min_length=10)
    conversation: List[Message]
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_turns: int = Field(default=16, ge=2, le=64)
    # 멀티 페르소나용 추가 파라미터
    active_persona_ids: List[str] = []                # 장면(Scene)에 포함할 인물들
    responders_this_turn: Optional[List[str]] = None  # 이번 턴 실제로 말할 인물(없으면 모델이 소수만 선택)


# ─────────────────────────────────────────────────────────────
# 라우팅
# ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_html():
    try:
        with open(HTML_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")

@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

@app.get("/personas")
async def list_personas():
    """UI에서 선택할 수 있도록(이미지/설명 포함) 페르소나 목록 제공"""
    return {
        "personas": [
            {k: p[k] for k in ("id", "name", "role", "image_url", "short_desc")}
            for p in PERSONAS
        ]
    }


@app.post("/verify-key")
async def verify_key(req: VerifyRequest):
    print(f"[/verify-key] inbound")  # ← 추가
    """사용자 입력 OpenAI 키 유효성 검증 (모델 목록 호출로 가볍게 확인)"""
    if OpenAI is None:
        raise HTTPException(status_code=500, detail="OpenAI SDK가 설치되어 있지 않거나 호환되지 않습니다. `pip install -U openai`")

    key = (req.api_key or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="API 키가 비어 있습니다.")

    try:
        client = OpenAI(api_key=key)
        _ = client.models.list()  # 호출 성공 == 키 유효
        return {"ok": True}
    except Exception as e:
        print("[/verify-key ERROR]", e.__class__.__name__, ":", str(e))
        traceback.print_exc()
        msg = str(e).lower()
        if "invalid api key" in msg or "incorrect api key" in msg or "authentication" in msg:
            raise HTTPException(status_code=401, detail="API 키가 올바르지 않습니다.")
        if "permission" in msg or "insufficient_quota" in msg:
            raise HTTPException(status_code=403, detail="권한/쿼터 문제입니다. 결제/한도를 확인하세요.")
        if "not found" in msg or "404" in msg:
            raise HTTPException(status_code=404, detail="엔드포인트/모델을 찾지 못했습니다.")
        if "rate limit" in msg or "429" in msg:
            raise HTTPException(status_code=429, detail="요청 한도를 초과했습니다.")
        if "ssl" in msg or "certificate" in msg:
            raise HTTPException(status_code=503, detail="SSL/인증서 문제로 연결 실패.")
        if "timed out" in msg or "timeout" in msg or "connection" in msg:
            raise HTTPException(status_code=503, detail="네트워크 연결 문제로 OpenAI API 접근 불가.")
        raise HTTPException(status_code=500, detail="알 수 없는 오류가 발생했습니다. 서버 로그를 확인하세요.")

@app.post("/chat")
async def chat(req: ChatRequest):
    """멀티 페르소나 장면 + 이번 턴 발화자 제어 + 한글 대화"""
    if OpenAI is None:
        raise HTTPException(status_code=500, detail="OpenAI SDK가 설치되어 있지 않거나 호환되지 않습니다.")

    key = (req.api_key or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="API 키가 비어 있습니다.")
    if not req.conversation:
        raise HTTPException(status_code=400, detail="대화 내용이 비어 있습니다.")

    client = OpenAI(api_key=key)

    # 페르소나 사전
    id2p: Dict[str, Dict] = {p["id"]: p for p in PERSONAS}

    # 장면(Scene)에 포함시킬 인물들
    scene_personas: List[Dict] = [id2p[i] for i in req.active_persona_ids if i in id2p]
    if not scene_personas:
        # 아무것도 선택 안 하면 기본으로 교사만
        scene_personas = [id2p["teacher"]]

    # 이번 턴 발화자 규칙
    if req.responders_this_turn:
        allowed_names = [id2p[i]["name"] for i in req.responders_this_turn if i in id2p]
        turn_rule = (
            "이번 턴에는 다음 인물만 말한다: " + ", ".join(f"[{n}]" for n in allowed_names) +
            ". 명시되지 않은 인물은 이번 턴에 발화하지 않는다."
        )
    else:
        turn_rule = (
            "이번 턴에는 상황에 맞는 소수의 인물만 말한다. "
            "모두가 동시에 말하지 않는다."
        )

    # 멀티 페르소나 시스템 프롬프트
    roster = "\n".join(
        f"- [{p['name']} / {p['role']}]: {p['short_desc']}"
        for p in scene_personas
    )
    persona_details = "\n".join(
        f"[{p['name']}] 규칙: {p['system_prompt']}"
        for p in scene_personas
    )
    SYSTEM_PROMPT = f"""너는 아래 등장인물들을 연기한다. 응답은 항상 한국어.
발화 형식: 각 발화는 반드시 `[이름]: 내용` 형태로 시작한다. (예: `[선생님]: 오늘 수업을 시작합니다.`)
말차례: 한 턴에 여러 명이 말할 수 있지만, 불필요하게 모두가 말하지 않는다.
길이: 불필요하게 길게 말하지 말고 필요한 정보에 집중한다.

[등장인물]
{roster}

[세부 규칙]
{persona_details}

[이번 턴 규칙]
{turn_rule}
"""

    # 최근 N턴 유지(+시스템 프롬프트 주입)
    tail = [m.model_dump() for m in req.conversation][-req.max_turns:]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + tail

    try:
        completion = client.chat.completions.create(
            model=req.model,
            messages=messages,
            temperature=req.temperature,
        )
        reply = completion.choices[0].message.content
        return JSONResponse({"reply": reply})
    except Exception as e:
        print(f"[ERROR] /chat: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="모델 호출 중 오류가 발생했습니다.")


if __name__ == "__main__":
    # 로컬 실행
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)



#uvicorn main1:app --reload
