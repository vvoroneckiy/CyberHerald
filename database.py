import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from datetime import datetime

# URL для асинхронного подключения к Postgres в докере
# Данные пользователя и пароля совпадают с твоим docker-compose.yml
DATABASE_URL = "postgresql+asyncpg://news_admin:secret_password_123@localhost:5432/news_db"

# Создаем движок БД
engine = create_async_engine(DATABASE_URL, echo=True)

# Фабрика сессий для обработки запросов
AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)

# Базовый класс для моделей таблиц
class Base(DeclarativeBase):
    pass

# Описываем, как будет выглядеть таблица новостей в Postgres
class DBNewsArticle(Base):
    __tablename__ = "news_articles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(nullable=False)
    source: Mapped[str] = mapped_column(default="Unknown")
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

# Функция-зависимость для получения сессии БД в эндпоинтах FastAPI
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session

# Функция автоматического создания таблиц при старте сервера
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)