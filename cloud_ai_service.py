from typing import Optional
from pydantic import BaseModel, Field


class NewsArticle(BaseModel):
    model_config = {"from_attributes": True}

    title: str = Field(..., description="Заголовок новости", examples=["Квантовый прорыв в России"])
    content: str = Field(..., description="Содержание новости", examples=["Группа учёных из МГУ..."])
    source: Optional[str] = Field("Unknown", description="Источник новости", examples=["ТАСС"])
    url: Optional[str] = Field(None, description="Ссылка на оригинал статьи", examples=["https://tass.ru/..."])


class Block(BaseModel):
    type: str = Field(..., description="Тип блока: header, text, chart, facts, sources", examples=["text"])
    data: dict = Field(..., description="Данные блока в зависимости от типа", examples=[{"title": "Заголовок", "content": "Текст новости", "source": "ТАСС"}])


class SearchResponse(BaseModel):
    user_text: str = Field(..., description="Распознанный текст запроса пользователя", examples=["Что нового в технологиях?"])
    blocks: list[Block] = Field(..., description="Список блоков с ответом (текст, графики, факты, источники)")
    audio_url: Optional[str] = Field(None, description="URL сгенерированного аудиоответа (если есть)", examples=[None])


class CloudAIService:
    TOPICS = {
        "футбол": [
            "футбол", "матч", "гол", "лига", "чемпионат", "спорт",
            "футболист", "тренер", "стадион", "мяч", "турнир",
        ],
        "политика": [
            "политика", "политику", "политике", "политикой", "политики",
            "политический", "политических", "политическом",
            "выборы", "президент", "правительство",
            "депутат", "госдума", "министр", "партия", "закон",
        ],
        "экономика": [
            "экономика", "экономику", "экономике", "экономикой",
            "экономический", "экономических",
            "финансы", "бюджет", "инфляция", "биржа",
            "валюта", "кризис", "рынок", "нефть", "газ", "курс",
        ],
        "технологии": [
            "технологии", "технологий", "технологиям", "технологиями",
            "технологиях", "технология", "технологию", "технологический",
            "it", "компьютер", "искусственный интеллект",
            "робот", "программирование", "gpt", "нейросеть",
            "инновации", "наука", "изобретение",
        ],
        "погода": [
            "погода", "погоду", "погоде", "погодой",
            "температура", "дождь", "снег", "ветер",
            "климат", "прогноз", "град", "жара",
        ],
        "здоровье": [
            "здоровье", "здоровья", "здоровью", "здоровьем",
            "болезнь", "лекарство", "врач", "больница",
            "вакцина", "медицина", "вирус", "операция",
        ],
        "культура": [
            "культура", "культуру", "культуре", "культурой",
            "культурный", "культурных", "культурном",
            "кино", "музыка", "выставка", "театр",
            "фильм", "концерт", "искусство", "книга", "фестиваль",
        ],
    }

    def _detect_topic(self, query: str) -> Optional[str]:
        query_lower = query.lower()
        for topic, keywords in self.TOPICS.items():
            if any(kw in query_lower for kw in keywords):
                return topic
        return None

    def _filter_articles(
        self, articles: list[NewsArticle], topic: Optional[str]
    ) -> list[NewsArticle]:
        if not topic:
            return articles[:5]
        keywords = self.TOPICS.get(topic, [])
        filtered = []
        for art in articles:
            text = f"{art.title} {art.content}".lower()
            if any(kw in text for kw in keywords):
                filtered.append(art)
        return filtered[:5]

    def _generate_chart_data(self, articles: list[NewsArticle]) -> Optional[dict]:
        if len(articles) < 2:
            return None
        source_counts = {}
        for art in articles:
            s = art.source or "Unknown"
            source_counts[s] = source_counts.get(s, 0) + 1
        if len(source_counts) >= 2:
            return {
                "chart_type": "bar",
                "title": "Количество новостей по источникам",
                "labels": list(source_counts.keys()),
                "values": list(source_counts.values()),
            }
        short_titles = [
            (art.title[:20] + "..." if len(art.title) > 20 else art.title)
            for art in articles[:5]
        ]
        return {
            "chart_type": "bar",
            "title": "Длина новостей (символы)",
            "labels": short_titles,
            "values": [len(art.content) for art in articles[:5]],
        }

    def _generate_facts(self, articles: list[NewsArticle]) -> list[dict]:
        if not articles:
            return []
        total = len(articles)
        avg_len = sum(len(a.content) for a in articles) // total
        sources = len(set(a.source for a in articles if a.source))
        return [
            {"label": "Всего новостей", "value": str(total)},
            {"label": "Средняя длина", "value": f"{avg_len} симв."},
            {"label": "Источников", "value": str(sources)},
        ]

    def generate_rag_answer(
        self, query: str, context_articles: list[NewsArticle], clean_query: Optional[str] = None
    ) -> SearchResponse:
        topic = self._detect_topic(clean_query or query)
        blocks = []

        if not context_articles:
            blocks.append(
                Block(type="header", data={"text": "Новостей по вашему запросу не найдено"})
            )
            blocks.append(
                Block(
                    type="text",
                    data={
                        "title": "Попробуйте изменить запрос",
                        "content": "К сожалению, не удалось найти новости по вашему запросу. Попробуйте спросить о другом.",
                        "source": "",
                    },
                )
            )
        else:
            filtered = self._filter_articles(context_articles, topic)
            articles = filtered if filtered else context_articles[:5]
            topic_name = topic if topic else "все темы"
            label = f"Новости по теме: {topic_name}" if filtered else f"Новости (похожие на тему: {topic_name})"
            blocks.append(
                Block(type="header", data={"text": label})
            )

            for art in articles:
                block_data = {
                    "title": art.title,
                    "content": art.content,
                    "source": art.source,
                }
                if art.url:
                    block_data["url"] = art.url
                blocks.append(Block(type="text", data=block_data))

            chart_data = self._generate_chart_data(articles)
            if chart_data:
                blocks.append(Block(type="chart", data=chart_data))

            facts = self._generate_facts(articles)
            if facts:
                blocks.append(Block(type="facts", data={"items": facts}))

            sources = list(set(art.source for art in articles if art.source))
            if sources:
                blocks.append(Block(type="sources", data={"items": sources}))

        return SearchResponse(user_text=query, blocks=blocks, audio_url=None)
