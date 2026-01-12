from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model_handler import ModelHandler


@pytest.fixture
def mock_llama_server_client():
    """Создает мок LlamaServerClient."""
    mock_client = MagicMock()
    mock_client.is_available.return_value = True
    mock_client._available = True

    async def check_health():
        return True

    async def generate_chat_completion_async(*args, **kwargs):
        if kwargs.get('stream', False):
            # Для streaming возвращаем генератор
            chunks = ['Тест', 'овый ', 'ответ']

            def stream_gen():
                for chunk in chunks:
                    yield chunk

            return stream_gen()
        # Для синхронного режима возвращаем строку
        return 'Тестовый ответ модели'

    mock_client.check_health = AsyncMock(side_effect=check_health)
    mock_client.generate_chat_completion = AsyncMock(side_effect=generate_chat_completion_async)

    async def close():
        pass

    mock_client.close = AsyncMock(side_effect=close)
    return mock_client


@pytest.fixture
def mock_model_handler(mock_llama_server_client):
    """Создает ModelHandler с мок-клиентом llama.cpp сервера."""
    with patch('app.model_handler.LlamaServerClient', return_value=mock_llama_server_client):
        handler = ModelHandler(
            llama_server_url='http://localhost:8080',
            system_prompt_type='assistant',
        )
        handler.client = mock_llama_server_client
        yield handler


@pytest.fixture
def mock_model_handler_for_api(mock_llama_server_client):
    """Создает ModelHandler с мок-клиентом для API тестов."""
    with patch('app.model_handler.LlamaServerClient', return_value=mock_llama_server_client):
        handler = ModelHandler(
            llama_server_url='http://localhost:8080',
            system_prompt_type='assistant',
        )
        handler.client = mock_llama_server_client
        return handler


@pytest.fixture
def test_client(mock_model_handler_for_api):
    """Создает тестовый клиент FastAPI."""
    # Подменяем реальный model_handler на мок
    app.state.model_handler = mock_model_handler_for_api
    return TestClient(app)


@pytest.fixture
def sample_messages():
    """Примеры сообщений для тестов."""
    return [
        {'role': 'user', 'content': 'Привет!'},
        {'role': 'assistant', 'content': 'Здравствуйте! Как дела?'},
        {'role': 'user', 'content': 'Хорошо, спасибо!'},
    ]


@pytest.fixture
def sample_single_message():
    """Одно сообщение для тестов."""
    return [{'role': 'user', 'content': 'Тестовое сообщение'}]
