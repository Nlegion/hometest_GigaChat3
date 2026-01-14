import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class LlamaServerClient:
    """Клиент для взаимодействия с llama.cpp сервером через HTTP API."""

    def __init__(self, server_url: str, timeout: float = 300.0) -> None:
        """Инициализирует клиент для llama.cpp сервера.

        Args:
            server_url: URL сервера (например, http://localhost:8080)
            timeout: Таймаут для HTTP запросов в секундах

        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        self._available = False

    async def check_health(self) -> bool:
        """Проверяет доступность llama.cpp сервера.

        Returns:
            True если сервер доступен, False иначе

        """
        function_name = 'LlamaServerClient.check_health'
        file_name = 'llama_server_client.py'
        try:
            # Попытка простого запроса к серверу
            response = await self.client.get(f'{self.server_url}/health', timeout=5.0)
            self._available = response.status_code == 200
            if self._available:
                logger.info('llama_server_available', function=function_name, file=file_name, server_url=self.server_url)
            else:
                logger.warning(
                    'llama_server_unavailable',
                    function=function_name,
                    file=file_name,
                    server_url=self.server_url,
                    status_code=response.status_code,
                )
            return self._available
        except httpx.RequestError as e:
            logger.warning('llama_server_health_check_failed', function=function_name, file=file_name, server_url=self.server_url, error=str(e))
            self._available = False
            return False
        except Exception as e:
            logger.exception('llama_server_health_check_exception', function=function_name, file=file_name, server_url=self.server_url, error=str(e))
            self._available = False
            return False

    def _format_messages_for_api(
        self, messages: list[dict[str, str]], system_prompt: str | None = None
    ) -> list[dict[str, str]]:
        """Форматирует сообщения для API llama.cpp сервера.

        Args:
            messages: Список сообщений в формате [{"role": "user|assistant", "content": "..."}]
            system_prompt: Опциональный системный промпт

        Returns:
            Отформатированный список сообщений для API

        """
        api_messages = []

        # Добавляем системный промпт, если указан
        if system_prompt:
            api_messages.append({'role': 'system', 'content': system_prompt})

        # Преобразуем сообщения в формат API
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            # Преобразуем роли в формат API
            if role == 'assistant':
                api_messages.append({'role': 'assistant', 'content': content})
            elif role == 'system':
                api_messages.append({'role': 'system', 'content': content})
            else:
                api_messages.append({'role': 'user', 'content': content})

        return api_messages

    async def generate_chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
        system_prompt: str | None = None,
        model_name: str = 'ai-sage/GigaChat3-10B-A1.8B',
    ) -> str | AsyncGenerator[str, None]:
        """Генерирует ответ через llama.cpp сервер.

        Args:
            messages: Список сообщений в формате [{"role": "user|assistant", "content": "..."}]
            temperature: Температура генерации (0.0-2.0)
            max_tokens: Максимальное количество токенов
            stream: Если True, возвращает генератор для потоковой генерации
            system_prompt: Опциональный системный промпт
            model_name: Имя модели для API

        Returns:
            Ответ модели (строка или генератор строк для streaming)

        Raises:
            RuntimeError: Если сервер недоступен или произошла ошибка

        """
        function_name = 'LlamaServerClient.generate_chat_completion'
        file_name = 'llama_server_client.py'
        if not self._available:
            # Проверяем доступность перед запросом
            available = await self.check_health()
            if not available:
                error_msg = f'llama.cpp сервер недоступен: {self.server_url}'
                logger.error('llama_server_not_available', function=function_name, file=file_name, server_url=self.server_url)
                raise RuntimeError(error_msg)

        api_messages = self._format_messages_for_api(messages, system_prompt)

        request_data: dict[str, Any] = {
            'model': model_name,
            'tool_choice': 'none',  # Важно для GigaChat3, чтобы избежать добавления системного промпта о функциях
            'messages': api_messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': stream,
        }

        if stream:
            return self._generate_stream_async(request_data)
        return await self._generate_sync(request_data)

    async def _generate_sync(self, request_data: dict[str, Any]) -> str:
        """Генерирует ответ синхронно через HTTP API."""
        function_name = 'LlamaServerClient._generate_sync'
        file_name = 'llama_server_client.py'
        try:
            response = await self.client.post(
                f'{self.server_url}/v1/chat/completions',
                json=request_data,
                headers={'Content-Type': 'application/json'},
            )
            response.raise_for_status()

            result = response.json()
            content = result['choices'][0]['message']['content']
            usage = result.get('usage', {})
            tokens_used = usage.get('total_tokens', 0)

            logger.info(
                'llama_server_generation_completed',
                function=function_name,
                file=file_name,
                tokens_used=tokens_used,
                response_length=len(content),
            )

            return content
        except httpx.HTTPStatusError as e:
            error_msg = f'Ошибка HTTP от llama.cpp сервера: {e.response.status_code} - {e.response.text}'
            logger.exception('llama_server_http_error', function=function_name, file=file_name, status_code=e.response.status_code, error=str(e))
            raise RuntimeError(error_msg) from e
        except httpx.RequestError as e:
            error_msg = f'Ошибка запроса к llama.cpp серверу: {e!s}'
            logger.exception('llama_server_request_error', function=function_name, file=file_name, error=str(e))
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f'Неожиданная ошибка при генерации: {e!s}'
            logger.exception('llama_server_generation_error', function=function_name, file=file_name, error=str(e))
            raise RuntimeError(error_msg) from e

    async def _generate_stream_async(self, request_data: dict[str, Any]) -> AsyncGenerator[str, None]:
        """Генерирует ответ потоково через HTTP API (async генератор)."""
        function_name = 'LlamaServerClient._generate_stream_async'
        file_name = 'llama_server_client.py'
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as async_client:
                async with async_client.stream(
                    'POST',
                    f'{self.server_url}/v1/chat/completions',
                    json=request_data,
                    headers={'Content-Type': 'application/json'},
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line or line == 'data: [DONE]':
                            continue

                        if line.startswith('data: '):
                            data_str = line[6:]  # Убираем префикс "data: "
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                logger.warning('llama_server_stream_parse_error', function=function_name, file=file_name, line=line)
                                continue

        except httpx.HTTPStatusError as e:
            error_msg = f'Ошибка HTTP при streaming: {e.response.status_code}'
            logger.exception('llama_server_stream_http_error', function=function_name, file=file_name, status_code=e.response.status_code)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f'Ошибка при потоковой генерации: {e!s}'
            logger.exception('llama_server_stream_error', function=function_name, file=file_name, error=str(e))
            raise RuntimeError(error_msg) from e

    async def close(self) -> None:
        """Закрывает HTTP клиент."""
        await self.client.aclose()

    def is_available(self) -> bool:
        """Проверяет, доступен ли сервер (по результатам последней проверки)."""
        return self._available
