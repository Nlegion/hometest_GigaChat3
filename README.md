# GigaChat3 Web Interface

Веб-интерфейс для тестирования модели GigaChat3-10B через FastAPI с поддержкой NVIDIA GPU.

## Описание

Проект предоставляет простой диалоговый веб-интерфейс для интерактивного тестирования модели GigaChat3-10B-A1.8B-GGUF (https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B-GGUF). Модель запускается через `llama-cpp-python` с поддержкой CUDA для ускорения на GPU.

## Возможности

- Веб-интерфейс для диалога с моделью
- Потоковая генерация ответов (streaming)
- Настройка параметров генерации (temperature, max_tokens)
- Выбор системных промптов (Помощник, Кодер, Аналитик ТЗ)
- Сохранение и загрузка истории диалога
- Структурированное логирование через structlog
- Полное покрытие тестами (unit + integration)

## Требования

- Python 3.12+
- NVIDIA GPU с поддержкой CUDA (для ускорения)
- Docker и Docker Compose (для контейнеризации)
- NVIDIA Container Toolkit (для GPU в Docker)

## Установка и запуск

### Локальный запуск

1. Установите зависимости:

```bash
pip install -r requirements.txt
```

Для разработки (с тестами):

```bash
pip install -r requirements-dev.txt
```

2. Убедитесь, что модель находится по пути `data/model/GigaChat3-10B-A1.8B-q6_k.gguf`

3. Создайте файл `.env` на основе `.env.example`:

```bash
cp .env.example .env
```

4. Запустите приложение:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Или напрямую:

```bash
python -m app.main
```

5. Откройте браузер и перейдите на `http://localhost:8000`

### Запуск в Docker

1. Убедитесь, что установлен NVIDIA Container Toolkit:

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

2. Создайте файл `.env` (опционально, можно использовать значения по умолчанию)

3. Запустите через docker-compose:

```bash
docker-compose up --build
```

4. Откройте браузер и перейдите на `http://localhost:8000`

### Сборка Docker образа вручную

```bash
docker build -t gigachat3-web .
docker run --gpus all -p 8000:8000 gigachat3-web
```

## Конфигурация

Все настройки можно изменить через переменные окружения в файле `.env`:

- `MODEL_PATH` - путь к модели GGUF (по умолчанию: `data/model/GigaChat3-10B-A1.8B-q6_k.gguf`)
- `N_GPU_LAYERS` - количество слоев для загрузки на GPU (по умолчанию: 40)
- `N_CTX` - размер контекста (по умолчанию: 8192)
- `HOST` - хост сервера (по умолчанию: 0.0.0.0)
- `PORT` - порт сервера (по умолчанию: 8000)
- `LOG_LEVEL` - уровень логирования (DEBUG, INFO, WARNING, ERROR)
- `LOG_FORMAT` - формат логов (json или console)
- `SYSTEM_PROMPT_TYPE` - тип системного промпта (assistant, coder, analyst)

## Тестирование

Запуск всех тестов:

```bash
pytest
```

С покрытием кода:

```bash
pytest --cov=app --cov-report=html
```

Запуск только unit тестов:

```bash
pytest tests/unit/
```

Запуск только integration тестов:

```bash
pytest tests/integration/
```

## API Endpoints

### `GET /`

Главная страница с веб-интерфейсом.

### `POST /api/chat`

Основной эндпоинт для диалога с моделью.

**Запрос:**
```json
{
  "messages": [
    {"role": "user", "content": "Привет!"}
  ],
  "temperature": 0.7,
  "max_tokens": 512,
  "stream": false,
  "system_prompt_type": "assistant"
}
```

**Ответ (обычный):**
```json
{
  "content": "Здравствуйте! Как дела?"
}
```

**Ответ (streaming):**
Server-Sent Events поток с чанками:
```
data: {"content": "Здрав"}
data: {"content": "ствуйте"}
data: [DONE]
```

### `GET /api/health`

Проверка статуса сервиса и модели.

**Ответ:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "loaded": true,
    "model_path": "data/model/GigaChat3-10B-A1.8B-q6_k.gguf",
    "n_gpu_layers": 40,
    "n_ctx": 8192,
    "system_prompt_type": "assistant"
  }
}
```

### `POST /api/reset`

Сброс контекста диалога.

**Ответ:**
```json
{
  "status": "ok",
  "message": "Контекст сброшен (в текущей реализации контекст не сохраняется)"
}
```

## Структура проекта

```
hometest_gigachat3/
├── app/
│   ├── core/
│   │   └── logging_config.py  # Настройка логирования
│   ├── main.py                # FastAPI приложение
│   ├── model_handler.py       # Обработчик модели
│   └── prompts.py             # Системные промпты
├── static/
│   ├── style.css              # Стили интерфейса
│   └── script.js               # Логика фронтенда
├── templates/
│   └── index.html             # Главная страница
├── tests/
│   ├── unit/                  # Unit тесты
│   └── integration/           # Integration тесты
├── data/
│   └── model/                 # Модель GGUF
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-dev.txt
```

## Логирование

Приложение использует структурированное логирование через `structlog`. Логи выводятся в формате JSON (production) или с цветным форматированием (development).

Пример лога:
```json
{
  "event": "generation_completed",
  "prompt_hash": "abc123",
  "tokens_used": 150,
  "generation_time_seconds": 2.5,
  "tokens_per_second": 60.0,
  "request_id": "uuid-here"
}
```

## Разработка

### Форматирование кода

```bash
ruff format .
ruff check --fix .
```

### Запуск линтера

```bash
ruff check .
```

## Устранение неполадок

### Модель не загружается

- Проверьте путь к модели в `MODEL_PATH`
- Убедитесь, что файл модели существует и доступен для чтения
- Проверьте логи на наличие ошибок загрузки

### GPU не используется

- Убедитесь, что установлен NVIDIA Container Toolkit (для Docker)
- Проверьте, что `N_GPU_LAYERS > 0`
- Проверьте доступность GPU: `nvidia-smi`

### Ошибки при сборке Docker образа

- Убедитесь, что используется правильный базовый образ CUDA
- Проверьте, что все зависимости установлены корректно
- Увеличьте время ожидания при сборке (модель может быть большой)

## Лицензия

Проект создан для тестирования модели GigaChat3-10B.
