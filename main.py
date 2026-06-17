import asyncio
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from cloud_ai_service import CloudAIService, NewsArticle, SearchResponse, Block
from voice_service import VoiceService
from database import init_db, get_db, DBNewsArticle
from qdrant_service import QdrantService

tags_metadata = [
    {
        "name": "System",
        "description": "Эндпоинты для проверки состояния сервера",
    },
    {
        "name": "News",
        "description": "Операции с новостями: получение списка и добавление новых статей",
    },
    {
        "name": "Voice",
        "description": "Голосовой поиск новостей с распознаванием речи и RAG-ответом",
    },
]

app = FastAPI(
    title="News AI Assistant API",
    description="Бэкенд новостного ИИ-ассистента с голосовым вводом и RAG-архитектурой",
    version="2.0",
    contact={
        "name": "CyberHerald Team",
        "url": "https://github.com/anomalyco/CyberHerald",
    },
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ai_service = CloudAIService()
voice_service = VoiceService()
qdrant_service = QdrantService()


class NewsCreateRequest(BaseModel):
    title: str = Field(
        ..., description="Заголовок новости", example="Российские учёные совершили прорыв в квантовых вычислениях"
    )
    content: str = Field(
        ..., description="Текст новости", example="Группа исследователей из МГУ представила первый в России 50-кубитный квантовый компьютер..."
    )
    source: str = Field(
        "Unknown", description="Источник новости", example="ТАСС"
    )


@app.on_event("startup")
async def on_startup():
    print("[INFO] Инициализация базы данных PostgreSQL...")
    await init_db()
    print("[INFO] База данных готова к работе!")

    print("[INFO] Инициализация Qdrant...")
    try:
        await qdrant_service.ensure_collection()
        print("[INFO] Qdrant готов к работе!")
    except Exception as e:
        print(f"[WARN] Qdrant недоступен при старте: {e}")


@app.get(
    "/",
    tags=["System"],
    summary="Проверка статуса сервера",
    description="Возвращает статус бэкенда и текущую версию API",
    responses={
        200: {
            "description": "Сервер работает",
            "content": {
                "application/json": {
                    "example": {"status": "CyberHerald Backend is running", "version": "2.0"}
                }
            },
        }
    },
)
def read_root():
    return {"status": "CyberHerald Backend is running", "version": "2.0"}


@app.get(
    "/api/v1/news",
    response_model=list[NewsArticle],
    tags=["News"],
    summary="Получить все новости",
    description="Возвращает список всех новостей из базы данных, отсортированных по дате создания (сначала новые)",
    responses={
        200: {"description": "Список новостей"},
        500: {"description": "Ошибка сервера при чтении новостей"},
    },
)
async def get_all_news(db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(
            select(DBNewsArticle).order_by(DBNewsArticle.created_at.desc())
        )
        return result.scalars().all()
    except Exception as e:
        print(f"[ERROR] Ошибка чтения ленты новостей: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/news/add",
    response_model=NewsArticle,
    tags=["News"],
    summary="Добавить новость",
    description="Добавляет новую новость в базу данных и асинхронно индексирует её векторное представление в Qdrant для поиска",
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Новость успешно добавлена"},
        500: {"description": "Ошибка сервера при добавлении новости"},
    },
)
async def add_news(
    article: NewsCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        db_article = DBNewsArticle(
            title=article.title,
            content=article.content,
            source=article.source,
        )
        db.add(db_article)
        await db.commit()
        await db.refresh(db_article)

        asyncio.create_task(
            qdrant_service.add_article(
                db_article.id,
                article.title,
                article.content,
                article.source,
            )
        )

        return db_article
    except Exception as e:
        print(f"[ERROR] Ошибка добавления новости: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/voice-query",
    response_model=SearchResponse,
    tags=["Voice"],
    summary="Голосовой поиск новостей",
    description="Принимает аудиофайл с голосовым запросом, распознаёт речь через Whisper, ищет релевантные новости в Qdrant и формирует RAG-ответ с блоками (текст, графики, факты, источники)",
    responses={
        200: {"description": "Успешный ответ на голосовой запрос"},
        400: {"description": "Файл не передан или пустой"},
        500: {"description": "Ошибка сервера при обработке голосового запроса"},
    },
)
async def process_voice_query(file: UploadFile = File(..., description="Аудиофайл с голосовым запросом (поддерживаются форматы: wav, mp3, ogg, m4a)")):
    temp_audio_path = f"temp_{file.filename}"

    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Файл не передан")

        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"[INFO] Распознавание: {temp_audio_path}")
        recognized_text = voice_service.speech_to_text(temp_audio_path)
        print(f"[INFO] Распознано: '{recognized_text}'")

        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        if not recognized_text.strip():
            return SearchResponse(
                user_text="[Звук не распознан]",
                blocks=[
                    Block(type="header", data={"text": "Не удалось распознать речь"}),
                    Block(
                        type="text",
                        data={
                            "title": "Попробуйте снова",
                            "content": "Извини, не удалось разобрать слова. Попробуй сказать четче.",
                            "source": "",
                        },
                    ),
                ],
                audio_url=None,
            )

        print("[INFO] Поиск контекста в Qdrant...")
        qdrant_results = await qdrant_service.search(recognized_text, limit=5)
        context_articles = [
            NewsArticle(title=r["title"], content=r["content"], source=r["source"])
            for r in qdrant_results
        ]

        print("[INFO] Генерация ответа...")
        result = ai_service.generate_rag_answer(recognized_text, context_articles)

        return result

    except Exception as e:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        print(f"[ERROR] voice-query: {e}")
        raise HTTPException(status_code=500, detail=str(e))