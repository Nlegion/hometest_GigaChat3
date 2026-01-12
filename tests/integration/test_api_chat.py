from unittest.mock import patch

import pytest


@pytest.fixture
def mock_model_handler_for_api(mock_model_handler, monkeypatch):
    """Подменяет глобальный model_handler в приложении."""
    with patch('app.main.model_handler', mock_model_handler):
        yield mock_model_handler


def test_api_chat_regular_response(test_client, mock_model_handler_for_api):
    """Тест обычного ответа через API."""
    response = test_client.post(
        '/api/chat',
        json={
            'messages': [{'role': 'user', 'content': 'Привет!'}],
            'stream': False,
            'temperature': 0.7,
            'max_tokens': 100,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert 'content' in data
    assert isinstance(data['content'], str)


def test_api_chat_streaming_response(test_client, mock_model_handler_for_api):
    """Тест потокового ответа через API."""
    response = test_client.post(
        '/api/chat',
        json={
            'messages': [{'role': 'user', 'content': 'Привет!'}],
            'stream': True,
            'temperature': 0.7,
            'max_tokens': 100,
        },
    )

    assert response.status_code == 200
    assert response.headers['content-type'] == 'text/event-stream; charset=utf-8'

    content = ''
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8') if isinstance(line, bytes) else line
            content += line_str + '\n'
            if '[DONE]' in line_str:
                break

    assert len(content) > 0


def test_api_chat_empty_messages(test_client, mock_model_handler_for_api):
    """Тест запроса с пустым списком сообщений."""
    response = test_client.post(
        '/api/chat',
        json={
            'messages': [],
            'stream': False,
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert 'detail' in data


def test_api_chat_model_not_loaded(test_client):
    """Тест запроса при незагруженной модели."""
    with patch('app.main.model_handler', None):
        response = test_client.post(
            '/api/chat',
            json={
                'messages': [{'role': 'user', 'content': 'Тест'}],
                'stream': False,
            },
        )

        assert response.status_code == 503
        data = response.json()
        assert 'detail' in data


def test_api_chat_custom_parameters(test_client, mock_model_handler_for_api):
    """Тест запроса с кастомными параметрами."""
    response = test_client.post(
        '/api/chat',
        json={
            'messages': [{'role': 'user', 'content': 'Тест'}],
            'stream': False,
            'temperature': 0.9,
            'max_tokens': 200,
            'system_prompt_type': 'coder',
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert 'content' in data


def test_api_chat_multiple_messages(test_client, mock_model_handler_for_api):
    """Тест запроса с несколькими сообщениями."""
    response = test_client.post(
        '/api/chat',
        json={
            'messages': [
                {'role': 'user', 'content': 'Вопрос 1'},
                {'role': 'assistant', 'content': 'Ответ 1'},
                {'role': 'user', 'content': 'Вопрос 2'},
            ],
            'stream': False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert 'content' in data


def test_api_chat_invalid_temperature(test_client, mock_model_handler_for_api):
    """Тест валидации параметра temperature."""
    response = test_client.post(
        '/api/chat',
        json={
            'messages': [{'role': 'user', 'content': 'Тест'}],
            'temperature': 3.0,
            'stream': False,
        },
    )

    assert response.status_code == 422


def test_api_chat_invalid_max_tokens(test_client, mock_model_handler_for_api):
    """Тест валидации параметра max_tokens."""
    response = test_client.post(
        '/api/chat',
        json={
            'messages': [{'role': 'user', 'content': 'Тест'}],
            'max_tokens': 10000,
            'stream': False,
        },
    )

    assert response.status_code == 422
