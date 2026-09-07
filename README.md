1. Клонировать репозиторий

git clone <ссылка-на-репозиторий>
cd <папка-проекта>

2. Настроить переменные окружения
Скопируйте шаблон и заполните своими значениями:

cp .env.example .env

3. Собрать образы

Для запуска на CPU:
docker compose build --no-cache

Для запуска на GPU (NVIDIA CUDA):
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build --no-cache rag-app

4. Запустить сервисы

CPU:
docker compose up

GPU:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up

При первом запуске сервис ollama-pull автоматически загрузит веса языковой модели, а приложение rag-app веса эмбеддинг-модели, реранкера и guardrail-модели. Это может занять продолжительное время в зависимости от скорости интернет-соединения.

5. Проверка работы

Откройте в браузере:

http://localhost:8000/docs

