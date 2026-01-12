from typing import Literal

SystemPromptType = Literal['assistant', 'coder', 'analyst']


SYSTEM_PROMPTS: dict[SystemPromptType, str] = {
    'assistant': (
        'Ты полезный помощник. Отвечай на вопросы четко и по делу. '
        'Используй понятный язык и структурируй ответы, когда это уместно.'
    ),
    'coder': (
        'Ты опытный программист. Помогай с написанием кода, отладкой и решением технических задач. '
        'Предоставляй примеры кода с комментариями и объяснениями.'
    ),
    'analyst': (
        'Ты аналитик технических заданий. Помогай в создании, проверке и улучшении ТЗ. '
        'Обращай внимание на полноту, четкость формулировок и структурированность документации.'
    ),
}


def format_chat_messages(
    messages: list[dict[str, str]],
    system_prompt: str | None = None,
) -> str:
    """Форматирует список сообщений в промпт для модели GigaChat3.

    GigaChat3 использует формат с разделителями <s> и </s> для сообщений.

    Args:
        messages: Список сообщений в формате [{"role": "user|assistant|system", "content": "..."}]
        system_prompt: Системный промпт (если не указан, берется из messages)

    Returns:
        Отформатированный промпт для модели

    """
    formatted_parts: list[str] = []

    if system_prompt:
        formatted_parts.append(f'<s>system\n{system_prompt}\n</s>')

    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')

        if role == 'system':
            if not system_prompt:
                formatted_parts.append(f'<s>system\n{content}\n</s>')
        elif role == 'user':
            formatted_parts.append(f'<s>user\n{content}\n</s>')
        elif role == 'assistant':
            formatted_parts.append(f'<s>assistant\n{content}\n</s>')

    formatted_parts.append('<s>assistant\n')

    return ''.join(formatted_parts)


def get_system_prompt(prompt_type: SystemPromptType = 'assistant') -> str:
    """Получает системный промпт по типу.

    Args:
        prompt_type: Тип промпта ('assistant', 'coder', 'analyst')

    Returns:
        Текст системного промпта

    """
    return SYSTEM_PROMPTS.get(prompt_type, SYSTEM_PROMPTS['assistant'])
