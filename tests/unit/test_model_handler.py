from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.model_handler import ModelHandler


def test_model_handler_init(mock_llama_server_client):
    """Тест инициализации ModelHandler."""
    with patch('app.model_handler.LlamaServerClient', return_value=mock_llama_server_client):
        handler = ModelHandler(
            llama_server_url='http://localhost:8080',
            system_prompt_type='assistant',
        )

    assert handler.client is not None
    assert handler.llama_server_url == 'http://localhost:8080'
    assert handler.system_prompt_type == 'assistant'


def test_model_handler_init_server_unavailable():
    """Тест инициализации с недоступным сервером."""
    mock_client = MagicMock()
    mock_client.check_health = AsyncMock(return_value=False)
    mock_client.is_available.return_value = False
    mock_client._available = False

    with patch('app.model_handler.LlamaServerClient', return_value=mock_client):
        handler = ModelHandler(
            llama_server_url='http://localhost:8080',
            system_prompt_type='assistant',
        )

    assert handler.client is None or not handler.is_loaded()


@pytest.mark.asyncio
async def test_generate_chat_response_sync(mock_model_handler, sample_single_message):
    """Тест синхронной генерации ответа."""
    # Настраиваем мок для синхронного ответа
    async def generate_sync(*args, **kwargs):
        return 'Тестовый ответ модели'

    mock_model_handler.client.generate_chat_completion = AsyncMock(side_effect=generate_sync)

    response = await mock_model_handler.generate_chat_response(
        messages=sample_single_message,
        temperature=0.7,
        max_tokens=100,
        stream=False,
    )

    assert isinstance(response, str)
    assert len(response) > 0
    mock_model_handler.client.generate_chat_completion.assert_called_once()


@pytest.mark.asyncio
async def test_generate_chat_response_stream(mock_model_handler, sample_single_message):
    """Тест потоковой генерации ответа."""
    # Мокаем _generate_stream напрямую как async генератор
    async def mock_generate_stream(*args, **kwargs):
        chunks = ['Тест', 'овый ', 'ответ']
        for chunk in chunks:
            yield chunk

    mock_model_handler._generate_stream = mock_generate_stream

    # generate_chat_response - async функция, которая возвращает async генератор
    # await-им coroutine, чтобы получить async генератор
    response_gen = await mock_model_handler.generate_chat_response(
        messages=sample_single_message,
        temperature=0.7,
        max_tokens=100,
        stream=True,
    )

    # response_gen - это async генератор, проверяем его
    assert hasattr(response_gen, '__aiter__'), 'Должен быть async генератор'
    
    # Итерируемся по генератору
    chunks = []
    async for chunk in response_gen:
        chunks.append(chunk)

    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert ''.join(chunks) == 'Тестовый ответ'


@pytest.mark.asyncio
async def test_generate_chat_response_model_not_loaded(mock_model_handler):
    """Тест генерации при недоступном сервере."""
    mock_model_handler.client = None

    with pytest.raises(RuntimeError, match='llama.cpp сервер недоступен'):
        await mock_model_handler.generate_chat_response(
            messages=[{'role': 'user', 'content': 'Тест'}],
            stream=False,
        )


def test_is_loaded(mock_model_handler):
    """Тест проверки доступности сервера."""
    assert mock_model_handler.is_loaded() is True

    mock_model_handler.client = None
    assert mock_model_handler.is_loaded() is False


def test_get_model_info(mock_model_handler):
    """Тест получения информации о модели."""
    info = mock_model_handler.get_model_info()

    assert info['loaded'] is True
    assert 'llama_server_url' in info
    assert 'model_name' in info
    assert 'system_prompt_type' in info
    assert 'use_gpu' in info


@pytest.mark.asyncio
async def test_generate_chat_response_parameters(mock_model_handler, sample_single_message):
    """Тест передачи параметров генерации."""
    async def generate_sync(*args, **kwargs):
        return 'Тестовый ответ'

    mock_model_handler.client.generate_chat_completion = AsyncMock(side_effect=generate_sync)

    await mock_model_handler.generate_chat_response(
        messages=sample_single_message,
        temperature=0.9,
        max_tokens=200,
        stream=False,
    )

    call_args = mock_model_handler.client.generate_chat_completion.call_args
    assert call_args is not None
    kwargs = call_args.kwargs
    assert kwargs.get('temperature') == 0.9
    assert kwargs.get('max_tokens') == 200
    assert kwargs.get('stream', False) is False


@pytest.mark.asyncio
async def test_generate_chat_response_exception_handling(mock_model_handler, sample_single_message):
    """Тест обработки исключений при генерации."""
    async def generate_with_error(*args, **kwargs):
        error_msg = 'Generation error'
        raise RuntimeError(error_msg)

    mock_model_handler.client.generate_chat_completion = AsyncMock(side_effect=generate_with_error)

    with pytest.raises(RuntimeError, match='Generation error'):
        await mock_model_handler.generate_chat_response(
            messages=sample_single_message,
            stream=False,
        )
