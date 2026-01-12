import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.logging_config import get_logger, setup_logging
from app.model_handler import ModelHandler
from app.prompts import SystemPromptType

logger = get_logger(__name__)


class Settings(BaseSettings):
    """Настройки приложения из переменных окружения."""

    # Параметры для llama.cpp сервера
    llama_server_url: str = Field(default='http://localhost:8080')
    use_llama_server: bool = Field(default=True)
    model_name: str = Field(default='ai-sage/GigaChat3-10B-A1.8B')

    # Старые параметры (для обратной совместимости, но не используются при use_llama_server=True)
    model_path: str = Field(default='data/model/GigaChat3-10B-A1.8B-q6_k.gguf')
    n_gpu_layers: int = Field(default=40)
    n_ctx: int = Field(default=8192)

    # Общие параметры
    host: str = Field(default='0.0.0.0')
    port: int = Field(default=8000)
    log_level: str = Field(default='INFO')
    log_format: str = Field(default='json')
    system_prompt_type: SystemPromptType = Field(default='assistant')

    model_config = SettingsConfigDict(
        env_file='.env', env_file_encoding='utf-8', case_sensitive=False
    )


settings = Settings()
model_handler: ModelHandler | None = None


class ChatMessage(BaseModel):
    """Модель сообщения в диалоге."""

    role: str = Field(..., description='Роль: user, assistant, system')
    content: str = Field(..., description='Содержимое сообщения')


class ChatRequest(BaseModel):
    """Модель запроса на генерацию ответа."""

    messages: list[ChatMessage] = Field(..., description='История диалога')
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description='Температура генерации')
    max_tokens: int = Field(
        default=512, ge=1, le=4096, description='Максимальное количество токенов'
    )
    stream: bool = Field(default=False, description='Потоковая генерация')
    system_prompt_type: SystemPromptType | None = Field(
        default=None,
        description='Тип системного промпта (assistant, coder, analyst)',
    )


class ChatResponse(BaseModel):
    """Модель ответа API."""

    content: str = Field(..., description='Сгенерированный ответ')
    tokens_used: int | None = Field(default=None, description='Использовано токенов')


class HealthResponse(BaseModel):
    """Модель ответа health check."""

    status: str = Field(..., description='Статус сервиса')
    model_loaded: bool = Field(..., description='Загружена ли модель')
    model_info: dict[str, Any] | None = Field(default=None, description='Информация о модели')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    global model_handler

    setup_logging(log_level=settings.log_level, log_format=settings.log_format)
    app_logger = get_logger(__name__)

    app_logger.info('application_starting', host=settings.host, port=settings.port)

    try:
        if settings.use_llama_server:
            model_handler = ModelHandler(
                llama_server_url=settings.llama_server_url,
                system_prompt_type=settings.system_prompt_type,
                model_name=settings.model_name,
            )
            app_logger.info(
                'application_started',
                model_loaded=model_handler.is_loaded(),
                llama_server_url=settings.llama_server_url,
            )
        else:
            # Старый способ через llama-cpp-python (для обратной совместимости)
            # ВНИМАНИЕ: Требует llama-cpp-python и не поддерживает DeepSeek2/GigaChat3
            app_logger.warning(
                'legacy_mode_disabled',
                message='Режим llama-cpp-python отключен. Используйте use_llama_server=true',
            )
            model_handler = None
    except Exception as e:
        app_logger.exception('application_startup_failed', error=str(e))
        model_handler = None

    yield

    # Закрываем клиент при завершении
    if model_handler and hasattr(model_handler, 'client') and model_handler.client:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(model_handler.client.close())
            finally:
                loop.close()
        except Exception as e:
            app_logger.warning('llama_server_client_close_error', error=str(e))

    app_logger.info('application_shutting_down')


app = FastAPI(
    title='GigaChat3 Web Interface',
    description='Веб-интерфейс для тестирования модели GigaChat3-10B',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Environment(
    loader=FileSystemLoader('templates'),
    autoescape=select_autoescape(['html', 'xml']),
)


@app.middleware('http')
async def log_requests(request: Request, call_next):
    """Middleware для логирования HTTP запросов."""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    logger.info(
        'request_started',
        method=request.method,
        path=request.url.path,
        query_params=dict(request.query_params),
    )

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        logger.info(
            'request_completed',
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            process_time_seconds=round(process_time, 3),
        )

        response.headers['X-Request-ID'] = request_id
        response.headers['X-Process-Time'] = str(round(process_time, 3))
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.exception(
            'request_failed',
            method=request.method,
            path=request.url.path,
            process_time_seconds=round(process_time, 3),
            error=str(e),
        )
        raise


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    """Главная страница с интерфейсом чата."""
    template = templates.get_template('index.html')
    return HTMLResponse(content=template.render(request=request))


@app.post('/api/chat', response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Основной эндпоинт для диалога с моделью.

    Поддерживает обычные и потоковые ответы.
    """
    if model_handler is None:
        error_msg = 'Модель не загружена'
        logger.error('model_not_available')
        raise HTTPException(status_code=503, detail=error_msg)

    if not request.messages:
        error_msg = 'Список сообщений не может быть пустым'
        logger.warning('empty_messages_request')
        raise HTTPException(status_code=400, detail=error_msg)

    try:
        system_prompt_type = request.system_prompt_type or settings.system_prompt_type
        if system_prompt_type != model_handler.system_prompt_type:
            model_handler.system_prompt_type = system_prompt_type

        messages_dict = [{'role': msg.role, 'content': msg.content} for msg in request.messages]

        if request.stream:
            return StreamingResponse(
                _stream_response(
                    model_handler, messages_dict, request.temperature, request.max_tokens
                ),
                media_type='text/event-stream',
            )
        response_text = await model_handler.generate_chat_response(
            messages=messages_dict,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=False,
        )

        if not isinstance(response_text, str):
            error_msg = 'Неожиданный тип ответа от модели'
            logger.error('unexpected_response_type')
            raise HTTPException(status_code=500, detail=error_msg)

        return ChatResponse(content=response_text)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception('chat_request_failed', error=str(e))
        raise HTTPException(status_code=500, detail=f'Ошибка при генерации ответа: {e!s}') from e


async def _stream_response(
    model_handler: ModelHandler,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
):
    """Генератор для потоковой отправки ответа."""
    try:
        response_gen = await model_handler.generate_chat_response(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        if not isinstance(response_gen, type(iter([]))):
            error_msg = 'Ожидался генератор для streaming'
            logger.error('invalid_stream_generator')
            yield f'data: {{"error": "{error_msg}"}}\n\n'
            return

        for chunk in response_gen:
            yield f'data: {{"content": {chunk!r}}}\n\n'

        yield 'data: [DONE]\n\n'
    except Exception as e:
        logger.exception('streaming_failed', error=str(e))
        yield f'data: {{"error": "Ошибка при потоковой генерации: {e!s}"}}\n\n'


@app.get('/api/health', response_model=HealthResponse)
async def health():
    """Проверка статуса сервиса и модели."""
    model_loaded = model_handler is not None and model_handler.is_loaded()
    model_info = model_handler.get_model_info() if model_handler else None

    # Добавляем информацию о режиме работы (GPU/CPU) в model_info
    if model_info and 'use_gpu' in model_info:
        model_info['mode'] = 'GPU' if model_info['use_gpu'] else 'CPU'

    return HealthResponse(
        status='healthy' if model_loaded else 'degraded',
        model_loaded=model_loaded,
        model_info=model_info,
    )


@app.post('/api/reset')
async def reset():
    """Сброс контекста диалога (заглушка для будущей реализации)."""
    logger.info('context_reset_requested')
    return {
        'status': 'ok',
        'message': 'Контекст сброшен (в текущей реализации контекст не сохраняется)',
    }


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        'app.main:app',
        host=settings.host,
        port=settings.port,
        log_config=None,
    )
