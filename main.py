import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

# Импортируем наши локальные сервисы
from cloud_ai_service import CloudAIService, NewsArticle
from voice_service import VoiceService
from database import init_db, get_db, DBNewsArticle

app = FastAPI(
    title="News AI Assistant API", 
    description="Бэкенд новостного ИИ-ассистента с голосовым вводом и RAG-архитектурой",
    version="1.0"
)

# НАСТРОЙКА CORS: Чтобы фронтенд мог слать запросы на сервер из браузера
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене лучше указать конкретный домен фронта
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализируем сервисы ИИ и Голоса
ai_service = CloudAIService()
voice_service = VoiceService()

# Схема ответа для фронтенда на голосовой запрос
class SearchResponse(BaseModel):
    user_text: str = Field(..., description="Распознанный текст из аудио через Whisper")
    ai_answer: str = Field(..., description="Сгенерированный ответ от облачной LLM")
    sources: list[str] = Field(..., description="Список источников/ссылок на новости, использованные в ответе")


# Событие старта приложения: автоматически создаем таблицы в Postgres, если их нет
@app.on_event("startup")
async def on_startup():
    print("[INFO] Инициализация базы данных PostgreSQL...")
    await init_db()
    print("[INFO] База данных готова к работе!")


@app.get("/")
def read_root():
    return {"status": "Backend is running", "environment": "Docker Container"}


# 1. ЭНДПОИНТ ДЛЯ ЛЕНТЫ НОВОСТЕЙ: ТЕПЕРЬ ТЯНЕТ ИЗ REAL POSTGRES
@app.get("/api/v1/news", response_model=list[NewsArticle])
async def get_all_news(db: AsyncSession = Depends(get_db)):
    """
    Возвращает список всех сохраненных новостей из базы данных PostgreSQL
    для отображения в ленте на фронтенде.
    """
    try:
        # Делаем асинхронный запрос в Postgres, сортируем по новизне
        result = await db.execute(select(DBNewsArticle).order_by(DBNewsArticle.created_at.desc()))
        db_news = result.scalars().all()
        
        # FastAPI сам смапит объекты SQLAlchemy в Pydantic-схему NewsArticle
        return db_news
    except Exception as e:
        print(f"[ERROR] Ошибка чтения ленты новостей: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка сервера при чтении БД: {str(e)}")


# 2. ГЛАВНЫЙ ЭНДПОИНТ: ПРИЕМ АУДИО ФАЙЛА + WHISPER + RAG (POSTGRES) + OLLAMA/QWEN
@app.post("/api/v1/voice-query", response_model=SearchResponse)
async def process_voice_query(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """
    Принимает аудиофайл (.mp3, .wav, .webm) с микрофона пользователя, 
    распознает его через Whisper, ищет контекст в Postgres и генерирует ответ через облачную LLM.
    """
    temp_audio_path = f"temp_{file.filename}"
    
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Файл не передан или имеет пустое имя")
            
        # 1. Сохраняем бинарный поток аудио во временный файл на диске контейнера
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Отправляем локальный файл в Whisper для перевода в текст
        print(f"[INFO] Начинаем распознавание файла через Whisper: {temp_audio_path}")
        recognized_text = voice_service.speech_to_text(temp_audio_path)
        print(f"[INFO] Whisper успешно распознал: '{recognized_text}'")
        
        # Чистим за собой временный файл, чтобы не забивать память докера
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
        if not recognized_text.strip():
            return SearchResponse(
                user_text="[Звук не распознан]",
                ai_answer="Извини, мне не удалось разобрать слова на аудиозаписи. Попробуй сказать четче.",
                sources=[]
            )
            
        # 3. Достаем новости из реальной базы данных для контекста RAG
        # В будущем здесь будет точечный поиск через Qdrant по эмбеддингам, 
        # а пока берем последние новости из Postgres для передачи в LLM
        db_result = await db.execute(select(DBNewsArticle).limit(10))
        db_articles = db_result.scalars().all()
        
        # Конвертируем модели БД в объекты NewsArticle для ИИ-сервиса
        context_articles = [
            NewsArticle(title=a.title, content=a.content, source=a.source) 
            for a in db_articles
        ]
        
        # 4. Передаем распознанный текст и реальный контекст в облачную Qwen
        print("[INFO] Запрос отправлен в Облачную LLM...")
        ai_answer = ai_service.generate_rag_answer(recognized_text, context_articles)
        
        # Собираем уникальные ссылки на источники новостей, которые пошли в контекст
        sources = list(set([art.source for art in context_articles if art.source]))
        
        return SearchResponse(
            user_text=recognized_text,
            ai_answer=ai_answer,
            sources=sources
        )
        
    except Exception as e:
        # Если произошла непредвиденная ошибка, обязательно удаляем временный файл
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        print(f"[ERROR] Ошибка в эндпоинте voice-query: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка бэкенд-сервера: {str(e)}")