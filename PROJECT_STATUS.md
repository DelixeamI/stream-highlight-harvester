# Текущее состояние проекта «Нарезка» (`narezka`)

**Дата актуализации:** Октябрь 2026 г.  
**Платформа:** Windows 11 (PowerShell), NVIDIA GeForce RTX 5070 (12 GB VRAM, sm_120, CUDA 13.3), Python 3.12 (`uv`).  
**Статус тестов:** **115 / 115 passed (100%)**, линтер `ruff` — 0 ошибок.

---

## 1. Где что находится (Краткая навигация)

* **Исходное техническое задание:** [`TZ_Narezka.md`](file:///d:/LLM%20PROJECT/Gemini/RU_STREAM_PARSER_V2/TZ_Narezka.md) (архитектура, формулы $S(t)$, схемы данных).
* **Сводные бенчмарки и реальные тайминги:** [`docs/benchmarks/BENCHMARKS.md`](file:///d:/LLM%20PROJECT/Gemini/RU_STREAM_PARSER_V2/docs/benchmarks/BENCHMARKS.md) (замеры из `meta.json` по стадиям и методология скорости).
* **Конфигурация по умолчанию:** [`config/default.yaml`](file:///d:/LLM%20PROJECT/Gemini/RU_STREAM_PARSER_V2/config/default.yaml) (профиль облачного VLM `gemini-3.1-flash-lite` и локального SLM `qwen2.5:14b`).
* **Безопасный REST API:** [`src/narezka/api/app.py`](file:///d:/LLM%20PROJECT/Gemini/RU_STREAM_PARSER_V2/src/narezka/api/app.py) (белый список каталогов VOD и расширений).
* **Модуль Video-VLM фреймов:** [`src/narezka/judge/frames.py`](file:///d:/LLM%20PROJECT/Gemini/RU_STREAM_PARSER_V2/src/narezka/judge/frames.py) (нарезка 8 кадров 360p JPEG base64).
* **Мультимодальный промпт VLM:** [`prompts/judge_vlm_ru.md`](file:///d:/LLM%20PROJECT/Gemini/RU_STREAM_PARSER_V2/prompts/judge_vlm_ru.md) (правила Visual Grounding и Personal Agency).
* **Учебное портфолио для собеседования:** [`INTERVIEW_PORTFOLIO.html`](file:///d:/LLM%20PROJECT/Gemini/RU_STREAM_PARSER_V2/INTERVIEW_PORTFOLIO.html) и [`INTERVIEW_PORTFOLIO.md`](file:///d:/LLM%20PROJECT/Gemini/RU_STREAM_PARSER_V2/INTERVIEW_PORTFOLIO.md).
* **Сводка результатов реальных прогонов:**
  * **CS2 VOD (34 мин):** 38 кандидатов $\to$ 32 отсеяно VLM $\to$ 6 чистых клипов (отчет [`projects/CS2_vod1_14b80e39/report.html`](file:///d:/LLM%20PROJECT/Gemini/RU_STREAM_PARSER_V2/projects/CS2_vod1_14b80e39/report.html)).
  * **Deadlock VOD (3 ч 08 мин):** 60 кандидатов $\to$ 50 отсеяно VLM $\to$ 10 эталонных клипов (отчет [`projects/Deadlock_vod1_cebb6641/report.html`](file:///d:/LLM%20PROJECT/Gemini/RU_STREAM_PARSER_V2/projects/Deadlock_vod1_cebb6641/report.html)).

---

## 2. Архитектура и реализованный DAG конвейера

Конвейер состоит из 14 последовательных детерминированных стадий с кешированием на диск (`sha256` от входных данных, кода и параметров):

| Стадия | Модуль | Описание | Статус |
| :--- | :--- | :--- | :---: |
| **S00_probe** | `stages/s00_probe.py` | Анализ метаданных VOD через ffprobe (разрешение, FPS, аудиодорожки) | ✅ Готово |
| **S01_audio** | `stages/s01_audio.py` | Быстрое извлечение 16 kHz mono аудиодорожки стрима | ✅ Готово |
| **S02_voice** | `stages/s02_voice.py` | Изоляция голоса стримера (HTDemucs: вокал vs игра/музыка) | ✅ Готово |
| **S03_vad_lid** | `stages/s03_vad_lid.py` | Silero VAD (детекция речи) + FastText LID (фильтрация не-RU) | ✅ Готово |
| **S04_asr** | `stages/s04_asr.py` | Распознавание речи через Faster-Whisper (large-v3-turbo) | ✅ Готово |
| **S06_audio_features** | `stages/s06_audio_features.py` | Акустические признаки (громкость LUFS, pitch Yin, смех, крики) | ✅ Готово |
| **S07_vision** | `stages/s07_vision.py` | **Компьютерное зрение (HUD & Killfeed)**: личные фраги, мультикиллы, смерти | ✅ Готово |
| **S08_chat_features** | `stages/s08_chat_features.py` | Анализ чата Twitch (плотность сообщений, спам смайлов LUL/KEKW) | ✅ Готово |
| **S09_semantic_scan** | `stages/s09_semantic_scan.py` | Семантический поиск эмоциональных реплик и панчлайнов | ✅ Готово |
| **S10_candidates** | `stages/s10_candidates.py` | Слияние кривых в интегральный интерес $S(t)$ и генерация окон кандидатов (+15-20 с подводки) | ✅ Готово |
| **S10.5_keyframes** | `judge/frames.py` | Извлечение ровно 8 кадров 360p JPEG base64 (~300 КБ payload, 0 MB VRAM) | ✅ Готово |
| **S11_judge** | `stages/s11_judge.py` | **Video-VLM Judge** (Gemini 3.1 Flash Lite) / Локальный SLM (`qwen2.5:14b`) | ✅ Готово |
| **S12_clips** | `stages/s12_clips.py` | Ранжирование кандидатов, диверсификация по категориям, сборка `report.html` | ✅ Готово |
| **S13_render** | `stages/s13_render.py` | Аппаратная нарезка MP4-клипов через NVENC (`h264_nvenc`) видеокарты | ✅ Готово |

---

## 3. Ключевые решения и прорывы системы

1. **Мультимодальный Video-VLM Судья (Gemini 3.1 Flash Lite):**
   * Преодоление **«Decibel Fallacy»**: классические аудио-детекторы реагируют на любой крик (например, падение стримера с лестницы на разминке CS2), порождая ложные хайлайты 9/10. Модуль `extract_candidate_keyframes()` извлекает 8 кадров 360p JPEG base64 (~300 КБ), а Gemini мгновенно видит разминку и выставляет 1/10 («Разминка перед матчем, фрагов нет»).
   * Преодоление **«Эффекта пассажира» (The Passenger Problem)** в Deadlock 6v6: промпт `judge_vlm_ru.md` ориентирует модель на **прицел стримера** и **нижний HP-бар**. Если игрок был пассивным участником боя, момент штрафуется до $\le 5$ (reject).
   * **Сетевая устойчивость:** прозрачный роутинг через локальный прокси Nova (`http://127.0.0.1:1371`, точка выхода Opera EU в Амстердаме) и двухуровневый retry с exponential backoff на 503/429.

2. **Компьютерное зрение ([`src/narezka/vision/`](file:///d:/LLM%20PROJECT/Gemini/RU_STREAM_PARSER_V2/src/narezka/vision)):**
   * **Counter-Strike 2 ([`cs2.py`](file:///d:/LLM%20PROJECT/Gemini/RU_STREAM_PARSER_V2/src/narezka/vision/cs2.py)):** Детектирует личные фраги по красной рамке Valve в киллфиде ($R \ge 135, R-G \ge 65, R-B \ge 75$), отслеживает сдвиг строк вниз, группирует фраги в серии (Double, Triple, Quad, Ace).
   * **Deadlock ([`deadlock.py`](file:///d:/LLM%20PROJECT/Gemini/RU_STREAM_PARSER_V2/src/narezka/vision/deadlock.py)):** Считывает верхний ростер из 12 героев ($S < 55$ при смерти), вспышки хитмаркеров и командные тимфайты.
   * **Быстродействие:** Сэмплирование 2 FPS на кропах экрана (0 MB VRAM, скорость ~10.6x от реального времени для стадии CV).

3. **Контекст и предыстория клипов:**
   * Перед каждым хайлайтом заложена подводка **15–20 секунд** (раскидка гранат, позиционирование, прицеливание), чтобы клип сохранял полную сюжетную завязку.

4. **Аппаратный экспорт видео:**
   * Быстрый рендеринг MP4 через `h264_nvenc` с CQ 21. Нарезка 10–20 клипов занимает секунды без нагрузки на CPU.

---

## 4. Как запускать проект

### Запуск локального сервера LLM (опционально, для локального профиля):
```powershell
ollama run qwen2.5:14b
```

### Запуск полного анализа стрима:
```powershell
# Для Counter-Strike 2:
uv run narezka run "D:\LLM PROJECT\Gemini\VODs\CS2_vod1.mp4" --game cs2 --until-stage s13_render

# Для Deadlock:
uv run narezka run "D:\LLM PROJECT\Gemini\VODs\Deadlock_vod1.mp4" --game deadlock --until-stage s13_render
```

### Запуск полного набора тестов:
```powershell
uv run pytest
```
