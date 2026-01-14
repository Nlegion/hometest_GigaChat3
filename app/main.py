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
            # Инициализируем клиент асинхронно
            await model_handler._init_client()
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
            await model_handler.client.close()
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
    # Логируем САМОЕ ПЕРВОЕ - до всех проверок
    request_id = str(uuid.uuid4())
    print(f'[DEBUG] CHAT_ENDPOINT_CALLED, request_id={request_id}')
    try:
        logger.info('CHAT_ENDPOINT_CALLED', request_id=request_id)
    except Exception as e:
        print(f'[DEBUG] ERROR LOGGING: {e}')
    
    # Логируем входящий запрос
    try:
        logger.info(
            'chat_request_received',
            request_id=request_id,
            messages_count=len(request.messages) if request.messages else 0,
            stream=request.stream,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_prompt_type=request.system_prompt_type,
        )
    except Exception as e:
        print(f'ERROR LOGGING chat_request_received: {e}')
    
    if model_handler is None:
        error_msg = 'Модель не загружена'
        logger.error('model_not_available', request_id=request_id)
        raise HTTPException(status_code=503, detail=error_msg)

    if not request.messages:
        error_msg = 'Список сообщений не может быть пустым'
        logger.warning('empty_messages_request', request_id=request_id)
        raise HTTPException(status_code=400, detail=error_msg)

    try:
        system_prompt_type = request.system_prompt_type or settings.system_prompt_type
        if system_prompt_type != model_handler.system_prompt_type:
            model_handler.system_prompt_type = system_prompt_type

        messages_dict = [{'role': msg.role, 'content': msg.content} for msg in request.messages]
        
        # Логируем подготовленные сообщения (без полного содержимого для безопасности)
        logger.info(
            'messages_prepared',
            request_id=request_id,
            messages_count=len(messages_dict),
            first_message_role=messages_dict[0]['role'] if messages_dict else None,
        )

        if request.stream:
            logger.info(
                'streaming_request_starting',
                request_id=request_id,
                messages_count=len(messages_dict),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            
            # _stream_response - это async генератор, передаем его напрямую
            # FastAPI StreamingResponse автоматически обработает async генератор
            print(f'[DEBUG] Creating stream_gen, request_id={request_id}')
            stream_gen = _stream_response(
                model_handler, messages_dict, request.temperature, request.max_tokens, request_id
            )
            print(f'[DEBUG] stream_gen created, type={type(stream_gen).__name__}, has_aiter={hasattr(stream_gen, "__aiter__")}')
            
            # Логируем тип объекта перед передачей в StreamingResponse
            logger.info(
                'stream_gen_created',
                request_id=request_id,
                stream_gen_type=type(stream_gen).__name__,
                has_aiter=hasattr(stream_gen, '__aiter__'),
                has_iter=hasattr(stream_gen, '__iter__'),
                is_coroutine=asyncio.iscoroutine(stream_gen),
            )
            
            print(f'[DEBUG] Returning StreamingResponse, request_id={request_id}')
            return StreamingResponse(stream_gen, media_type='text/event-stream')
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
    request_id: str | None = None,
):
    """Генератор для потоковой отправки ответа."""
    print(f'[DEBUG] _stream_response called, request_id={request_id}')
    try:
        print(f'[DEBUG] _stream_response: logging start, request_id={request_id}')
        logger.info(
            '_stream_response_started',
            request_id=request_id,
            messages_count=len(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        print(f'[DEBUG] _stream_response: logged start, request_id={request_id}')
        
        # generate_chat_response - async функция, при stream=True она возвращает coroutine от async генератора
        # Нужно await-ить coroutine, чтобы получить async генератор
        logger.info('calling_generate_chat_response', request_id=request_id)
        try:
            # Вызываем generate_chat_response с await, чтобы получить coroutine от async генератора
            response_gen_coro = await model_handler.generate_chat_response(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            logger.info('generate_chat_response_coro_received', request_id=request_id, coro_type=type(response_gen_coro).__name__)
            
            # await-им coroutine от async генератора, чтобы получить сам генератор
            response_gen = await response_gen_coro
            logger.info(
                'generate_chat_response_completed',
                request_id=request_id,
                response_received=True,
                response_type=type(response_gen).__name__,
            )
        except Exception as e:
            logger.exception(
                'generate_chat_response_failed',
                request_id=request_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            import json
            error_json = json.dumps({'error': f'Ошибка при получении генератора: {e!s}'}, ensure_ascii=False)
            yield f'data: {error_json}\n\n'
            return

        # Логируем тип для диагностики
        response_type_name = type(response_gen).__name__
        has_aiter = hasattr(response_gen, '__aiter__')
        has_iter = hasattr(response_gen, '__iter__')
        logger.info(
            'stream_generator_received',
            request_id=request_id,
            response_type=response_type_name,
            has_aiter=has_aiter,
            has_iter=has_iter,
            module=type(response_gen).__module__,
        )

        # Проверяем, что это async генератор
        if not has_aiter:
            error_msg = f'Ожидался async генератор для streaming, получен {response_type_name}'
            logger.error(
                'invalid_stream_generator',
                request_id=request_id,
                response_type=response_type_name,
                has_aiter=has_aiter,
                has_iter=has_iter,
            )
            import json
            error_json = json.dumps({'error': error_msg}, ensure_ascii=False)
            yield f'data: {error_json}\n\n'
            return
        
        logger.info('starting_async_iteration', request_id=request_id)

        chunks_count = 0
        async for chunk in response_gen:
            if chunk:
                chunks_count += 1
                # Экранируем JSON правильно
                import json
                chunk_json = json.dumps({'content': chunk}, ensure_ascii=False)
                yield f'data: {chunk_json}\n\n'
                
                # Логируем первые несколько чанков для диагностики
                if chunks_count <= 3:
                    logger.debug('chunk_yielded', request_id=request_id, chunk_number=chunks_count, chunk_length=len(chunk))

        logger.info('streaming_completed', request_id=request_id, total_chunks=chunks_count)
        yield 'data: [DONE]\n\n'
    except Exception as e:
        logger.exception('streaming_failed', request_id=request_id, error=str(e), error_type=type(e).__name__)
        import json
        error_json = json.dumps({'error': f'Ошибка при потоковой генерации: {e!s}'}, ensure_ascii=False)
        yield f'data: {error_json}\n\n'


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
