from app.prompts import SystemPromptType, format_chat_messages, get_system_prompt


def test_get_system_prompt_assistant():
    """Тест получения системного промпта типа assistant."""
    prompt = get_system_prompt('assistant')
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_get_system_prompt_coder():
    """Тест получения системного промпта типа coder."""
    prompt = get_system_prompt('coder')
    assert isinstance(prompt, str)
    assert 'программист' in prompt.lower() or 'код' in prompt.lower()


def test_get_system_prompt_analyst():
    """Тест получения системного промпта типа analyst."""
    prompt = get_system_prompt('analyst')
    assert isinstance(prompt, str)
    assert 'аналитик' in prompt.lower() or 'тз' in prompt.lower()


def test_get_system_prompt_default():
    """Тест получения промпта по умолчанию при неверном типе."""
    from typing import cast

    prompt = get_system_prompt(cast('SystemPromptType', 'invalid_type'))
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_format_chat_messages_single_user():
    """Тест форматирования одного пользовательского сообщения."""
    messages = [{'role': 'user', 'content': 'Привет!'}]
    result = format_chat_messages(messages)
    assert '<s>user\nПривет!\n</s>' in result
    assert result.endswith('<s>assistant\n')


def test_format_chat_messages_user_assistant():
    """Тест форматирования диалога пользователь-ассистент."""
    messages = [
        {'role': 'user', 'content': 'Вопрос'},
        {'role': 'assistant', 'content': 'Ответ'},
    ]
    result = format_chat_messages(messages)
    assert '<s>user\nВопрос\n</s>' in result
    assert '<s>assistant\nОтвет\n</s>' in result
    assert result.endswith('<s>assistant\n')


def test_format_chat_messages_with_system_prompt():
    """Тест форматирования с системным промптом."""
    messages = [{'role': 'user', 'content': 'Тест'}]
    system_prompt = 'Ты помощник'
    result = format_chat_messages(messages, system_prompt=system_prompt)
    assert '<s>system\nТы помощник\n</s>' in result
    assert '<s>user\nТест\n</s>' in result


def test_format_chat_messages_with_system_role():
    """Тест форматирования с сообщением role=system."""
    messages = [
        {'role': 'system', 'content': 'Системное сообщение'},
        {'role': 'user', 'content': 'Пользовательское сообщение'},
    ]
    result = format_chat_messages(messages)
    assert '<s>system\nСистемное сообщение\n</s>' in result
    assert '<s>user\nПользовательское сообщение\n</s>' in result


def test_format_chat_messages_multiple_turns():
    """Тест форматирования многоходового диалога."""
    messages = [
        {'role': 'user', 'content': 'Вопрос 1'},
        {'role': 'assistant', 'content': 'Ответ 1'},
        {'role': 'user', 'content': 'Вопрос 2'},
        {'role': 'assistant', 'content': 'Ответ 2'},
    ]
    result = format_chat_messages(messages)
    assert result.count('<s>user\n') == 2
    assert result.count('<s>assistant\n') == 3
    assert result.endswith('<s>assistant\n')


def test_format_chat_messages_empty_content():
    """Тест форматирования сообщения с пустым содержимым."""
    messages = [{'role': 'user', 'content': ''}]
    result = format_chat_messages(messages)
    assert '<s>user\n\n</s>' in result
