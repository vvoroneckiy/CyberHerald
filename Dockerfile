# Базовый образ с Python 3.12 на легком Linux (Slim)
FROM python:3.12-slim

# Ставим системные утилиты для работы со звуком (нужны для Whisper)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Указываем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл зависимостей (сейчас мы его создадим)
COPY requirements.txt .

# Ставим библиотеки прямо внутрь контейнера (здесь venv не нужен, контейнер уже изолирован)
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь наш код из папки проекта в контейнер
COPY . .

# Открываем порт 8000 наружу
EXPOSE 8000

# Команда для запуска сервера внутри докера
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]