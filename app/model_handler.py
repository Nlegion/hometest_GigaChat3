import asyncio
import hashlib
import time
from collections.abc import AsyncGenerator, Generator

from app.core.logging_config import get_logger
from app.llama_server_client import LlamaServerClient
from app.prompts import SystemPromptType, get_system_prompt

logger = get_logger(__name__)




class ModelHandler:
    """Обработчик для работы с моделью GigaChat3 через llama.cpp сервер."""

    def __init__(
        self,
        llama_server_url: str = 'http://localhost:8080',
        system_prompt_type: SystemPromptType = 'assistant',
        model_name: str = 'ai-sage/GigaChat3-10B-A1.8B',
    ) -> None:
        """Инициализирует обработчик модели.

        Args:
            llama_server_url: URL llama.cpp сервера
            system_prompt_type: Тип системного промпта
            model_name: Имя модели для API

        """
        self.llama_server_url = llama_server_url
        self.system_prompt_type = system_prompt_type
        self.model_name = model_name
        self.client: LlamaServerClient | None = None
        self.use_gpu: bool = True  # Предполагаем GPU, так как сервер обычно на GPU

        # Создаем клиент (без проверки health, это будет сделано асинхронно)
        try:
            self.client = LlamaServerClient(self.llama_server_url)
        except Exception as e:
            logger.exception('model_init_exception', llama_server_url=llama_server_url, error=str(e))
            self.client = None
            self.use_gpu = False

    async def _init_client(self) -> None:
        """Инициализирует клиент для llama.cpp сервера (асинхронно).

        При ошибке клиент остается None, исключение не выбрасывается.
        """
        if self.client is None:
            return

        start_time = time.time()

        logger.info(
            'llama_server_client_init_started',
            llama_server_url=self.llama_server_url,
            model_name=self.model_name,
        )

        # Проверяем доступность сервера
        try:
            available = await self.client.check_health()

            if available:
                init_time = time.time() - start_time
                logger.info(
                    'llama_server_client_init_success',
                    llama_server_url=self.llama_server_url,
                    init_time_seconds=round(init_time, 2),
                )
            else:
                logger.warning(
                    'llama_server_client_unavailable',
                    llama_server_url=self.llama_server_url,
                )
                self.client = None
                self.use_gpu = False
        except Exception as e:
            error_msg = f'Ошибка при инициализации клиента llama.cpp сервера: {e!s}'
            logger.exception('llama_server_client_init_exception', error=str(e))
            self.client = None
            self.use_gpu = False
            logger.warning('llama_server_client_init_failed_graceful', error=error_msg)

    async def generate_chat_response(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """Генерирует ответ модели на основе истории диалога.

        Args:
            messages: Список сообщений в формате [{"role": "user|assistant", "content": "..."}]
            temperature: Температура генерации (0.0-2.0)
            max_tokens: Максимальное количество токенов в ответе
            stream: Если True, возвращает генератор для потоковой генерации

        Returns:
            Ответ модели (строка или генератор строк для streaming)

        """
        if self.client is None:
            error_msg = 'llama.cpp сервер недоступен'
            logger.error('llama_server_not_available')
            raise RuntimeError(error_msg)

        start_time = time.time()
        prompt_length = sum(len(msg.get('content', '')) for msg in messages)

        system_prompt = get_system_prompt(self.system_prompt_type)

        # Создаем хэш для логирования
        messages_str = str(messages) + str(system_prompt)
        prompt_hash = hashlib.sha256(messages_str.encode()).hexdigest()[:16]

        logger.info(
            'generation_started',
            prompt_hash=prompt_hash,
            prompt_length=prompt_length,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            system_prompt_type=self.system_prompt_type,
        )

        try:
            if stream:
                logger.info('generate_chat_response_returning_stream', prompt_hash=prompt_hash)
                # _generate_stream() возвращает async генератор напрямую (не coroutine)
                # Возвращаем генератор напрямую
                return self._generate_stream(
                    messages, system_prompt, temperature, max_tokens, prompt_hash, start_time
                )
            return await self._generate_sync(
                messages, system_prompt, temperature, max_tokens, prompt_hash, start_time
            )
        except Exception as e:
            generation_time = time.time() - start_time
            logger.exception(
                'generation_failed',
                prompt_hash=prompt_hash,
                prompt_length=prompt_length,
                generation_time_seconds=round(generation_time, 2),
                error=str(e),
            )
            raise

    async def _generate_sync(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        prompt_hash: str,
        start_time: float,
    ) -> str:
        """Генерирует ответ синхронно через HTTP API."""
        if self.client is None:
            raise RuntimeError('llama.cpp сервер недоступен')

        # Вызываем async метод напрямую, так как мы в async контексте
        generated_text = await self.client.generate_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            system_prompt=system_prompt,
            model_name=self.model_name,
        )

        generation_time = time.time() - start_time
        # Оценка токенов (примерная, так как API может не возвращать usage)
        estimated_tokens = len(generated_text.split())
        tokens_per_second = round(estimated_tokens / generation_time, 2) if generation_time > 0 else 0

        logger.info(
            'generation_completed',
            prompt_hash=prompt_hash,
            estimated_tokens=estimated_tokens,
            generation_time_seconds=round(generation_time, 2),
            tokens_per_second=tokens_per_second,
            response_length=len(generated_text),
        )

        return generated_text

    async def _generate_stream(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        prompt_hash: str,
        start_time: float,
    ):
        """Генерирует ответ потоково через HTTP API (async генератор)."""
        if self.client is None:
            raise RuntimeError('llama.cpp сервер недоступен')

        # Получаем генератор от клиента (async метод возвращает генератор для streaming, не нужно await)
        stream_gen = self.client.generate_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            system_prompt=system_prompt,
            model_name=self.model_name,
        )
        
        # generate_chat_completion - это async функция, нужно await чтобы получить генератор
        stream_gen = await stream_gen

        tokens_count = 0
        full_text = ''

        async for chunk in stream_gen:
            if chunk:
                full_text += chunk
                tokens_count += 1
                yield chunk

        generation_time = time.time() - start_time
        tokens_per_second = round(tokens_count / generation_time, 2) if generation_time > 0 else 0

        logger.info(
            'streaming_completed',
            prompt_hash=prompt_hash,
            tokens_generated=tokens_count,
            generation_time_seconds=round(generation_time, 2),
            tokens_per_second=tokens_per_second,
            response_length=len(full_text),
        )

    def is_loaded(self) -> bool:
        """Проверяет, доступен ли llama.cpp сервер."""
        return self.client is not None and self.client.is_available()

    def get_model_info(self) -> dict[str, str | int | bool]:
        """Возвращает информацию о модели.

        Returns:
            Словарь с информацией о модели

        """
        return {
            'loaded': self.is_loaded(),
            'llama_server_url': self.llama_server_url,
            'model_name': self.model_name,
            'system_prompt_type': self.system_prompt_type,
            'use_gpu': self.use_gpu,
        }
