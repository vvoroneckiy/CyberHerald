import os
from faster_whisper import WhisperModel

class VoiceService:
    def __init__(self):
        # Выбираем модель. 'tiny' или 'base' идеально подходят для обычного ПК/сервера
        # Загрузка будет происходить на CPU, если нет мощной GPU под рукой
        self.model_size = "tiny"
        print(f"[INFO] Загрузка голосовой модели Whisper ({self.model_size})...")
        self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        print("[INFO] Модель Whisper успешно загружена!")

    def speech_to_text(self, audio_file_path: str) -> str:
        """Принимает путь к файлу записи и возвращает текст"""
        if not os.path.exists(audio_file_path):
            print(f"[ERROR] Файл {audio_file_path} не найден.")
            return ""
            
        try:
            # Прерываем аудио на сегменты и склеиваем текст
            segments, info = self.model.transcribe(audio_file_path, beam_size=5, language="ru")
            
            recognized_text = "".join([segment.text for segment in segments])
            return recognized_text.strip()
            
        except Exception as e:
            print(f"[ERROR] Ошибка распознавания речи: {e}")
            return ""