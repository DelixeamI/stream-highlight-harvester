"""Command-line interface for the Narezka application using Typer and Rich."""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import soundfile as sf
import typer
from rich.console import Console

from narezka import __version__
from narezka.asr.base import SpeechSegment
from narezka.asr.factory import get_asr_backend
from narezka.config import load_config
from narezka.doctor import run_doctor
from narezka.eval.asr_eval import evaluate_gold_directory, print_evaluation_report
from narezka.eval.highlights_eval import (
    evaluate_highlights,
    load_ground_truth,
    print_highlights_report,
)
from narezka.export.csv_export import export_moments_csv, export_youtube_timestamps
from narezka.export.json_export import export_moments_json
from narezka.export.report_html import render_html_report
from narezka.logging_setup import setup_logging
from narezka.media.render import render_clips_batch
from narezka.models import Candidate, MomentsDocument
from narezka.pipeline.dag import PipelineDAG
from narezka.pipeline.stages.s00_probe import S00ProbeStage
from narezka.pipeline.stages.s01_audio_extract import S01AudioExtractStage
from narezka.pipeline.stages.s02_voice_isolation import S02VoiceIsolationStage
from narezka.pipeline.stages.s03_vad_lid import S03VadLidStage
from narezka.pipeline.stages.s04_asr import S04AsrStage
from narezka.pipeline.stages.s06_audio_features import S06AudioFeaturesStage
from narezka.pipeline.stages.s07_vision import S07VisionStage
from narezka.pipeline.stages.s08_chat_features import S08ChatFeaturesStage
from narezka.pipeline.stages.s09_semantic_scan import S09SemanticScanStage
from narezka.pipeline.stages.s10_candidates import S10CandidatesStage
from narezka.pipeline.stages.s11_judge import S11JudgeStage
from narezka.pipeline.stages.s12_clips import S12ClipsStage
from narezka.pipeline.stages.s13_render import S13RenderStage
from narezka.project import Project

if sys.platform == "win32":
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(
    name="narezka",
    help="Локальный ИИ-агент поиска хайлайтов в VOD игровых стримов с русской речью.",
    no_args_is_help=True,
)
eval_app = typer.Typer(name="eval", help="Инструменты оценки качества (WER/CER, recall, precision).")
app.add_typer(eval_app, name="eval")

console = Console(force_terminal=True)
logger = logging.getLogger("narezka.cli")


def build_pipeline_dag() -> PipelineDAG:
    """Construct complete DAG with all pipeline stages (S00 through S13)."""
    dag = PipelineDAG()
    dag.register(S00ProbeStage())
    dag.register(S01AudioExtractStage())
    dag.register(S02VoiceIsolationStage())
    dag.register(S03VadLidStage())
    dag.register(S04AsrStage())
    dag.register(S06AudioFeaturesStage())
    dag.register(S07VisionStage())
    dag.register(S08ChatFeaturesStage())
    dag.register(S09SemanticScanStage())
    dag.register(S10CandidatesStage())
    dag.register(S11JudgeStage())
    dag.register(S12ClipsStage())
    dag.register(S13RenderStage())
    return dag


def version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold cyan]narezka[/bold cyan] версия {__version__}")
        raise typer.Exit(code=0)


@app.callback()
def main(
    version: bool | None = typer.Option(
        None,
        "--version",
        "-v",
        help="Показать версию приложения и выйти.",
        callback=version_callback,
        is_eager=True,
    ),
    debug: bool = typer.Option(False, "--debug", help="Включить подробный отладочный вывод (DEBUG)."),
) -> None:
    """Глобальный обработчик CLI."""
    setup_logging(debug=debug)


@app.command(name="doctor")
def doctor_cmd(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Путь к пользовательскому конфигурационному файлу (например, config/local.yaml).",
    ),
) -> None:
    """Проверить готовность окружения, видеокарты (RTX 5070 / sm_120), FFmpeg и зависимостей."""
    try:
        cfg = load_config(config_path=config)
    except Exception as exc:
        console.print(f"[bold red]Ошибка загрузки конфигурации:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    ok = run_doctor(cfg)
    if not ok:
        raise typer.Exit(code=2)
    raise typer.Exit(code=0)


@app.command(name="analyze")
def analyze_cmd(
    vod: Path = typer.Argument(..., help="Путь к файлу VOD (mp4, mkv, flv)."),
    chat: Path | None = typer.Option(None, "--chat", help="Путь к JSON-файлу чата Twitch."),
    game: str | None = typer.Option(None, "--game", "-g", help="Название игры (Counter-Strike 2, Deadlock, Dota 2 и др.)."),
    hero: str | None = typer.Option(None, "--hero", help="Имя героя стримера (для Deadlock)."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Путь к файлу конфигурации."),
    until_stage: str | None = typer.Option(
        None, "--until-stage", help="Остановить выполнение после указанной стадии."
    ),
    force_stage: str | None = typer.Option(
        None, "--force-stage", help="Принудительно перезапустить стадию, игнорируя кеш."
    ),
) -> None:
    """Запустить пайплайн анализа VOD стрима."""
    if not vod.is_file():
        console.print(f"[bold red]Ошибка:[/bold red] Файл VOD не найден: {vod}")
        raise typer.Exit(code=1)

    overrides = {}
    if chat:
        if not chat.is_file():
            console.print(f"[bold red]Ошибка:[/bold red] Файл чата не найден: {chat}")
            raise typer.Exit(code=1)
        overrides["chat"] = {"path": str(chat.resolve())}

    # Auto-detect game if not provided explicitly
    detected_game = game
    if not detected_game:
        stem = vod.stem.lower()
        if "cs2" in stem or "cs_go" in stem or "cs" in stem or "counter" in stem:
            detected_game = "Counter-Strike 2"
        elif "deadlock" in stem:
            detected_game = "Deadlock"
        elif "dota" in stem:
            detected_game = "Dota 2"
    if detected_game:
        overrides["game"] = detected_game

    cfg = load_config(config_path=config, overrides=overrides if overrides else None)
    project = Project.create(vod_path=vod, config=cfg)

    # Copy chat to project dir if provided
    if chat and chat.is_file():
        import shutil

        target_chat = project.project_dir / "chat.json"
        if not target_chat.is_file() or target_chat.resolve() != chat.resolve():
            shutil.copy2(chat, target_chat)

    dag = build_pipeline_dag()
    target_until = until_stage or "s12_clips"

    try:
        dag.run(project, until_stage=target_until, force_stage=force_stage)
        console.print(f"\n[bold green][OK] Анализ успешно завершён![/bold green] Проект: [cyan]{project.project_dir}[/cyan]")

        # Print outputs if reached s12
        report_file = project.project_dir / "report.html"
        if report_file.is_file():
            console.print(f"Отчёт HTML: [cyan]{report_file}[/cyan]")
            console.print(f"Моменты JSON: [cyan]{project.project_dir / 'moments.json'}[/cyan]")
            console.print(f"Моменты CSV: [cyan]{project.project_dir / 'moments.csv'}[/cyan]")
            console.print(f"Таймкоды YouTube: [cyan]{project.project_dir / 'youtube_timestamps.txt'}[/cyan]")
    except Exception as exc:
        console.print(f"[bold red]Ошибка выполнения пайплайна:[/bold red] {exc}")
        raise typer.Exit(code=3) from exc

    raise typer.Exit(code=0)


@app.command(name="run")
def run_cmd(
    vod: Path = typer.Argument(..., help="Путь к файлу VOD (mp4, mkv, flv)."),
    chat: Path | None = typer.Option(None, "--chat", help="Путь к JSON-файлу чата Twitch."),
    game: str | None = typer.Option(None, "--game", "-g", help="Название игры (Counter-Strike 2, Deadlock, Dota 2 и др.)."),
    hero: str | None = typer.Option(None, "--hero", help="Имя героя стримера (для Deadlock)."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Путь к файлу конфигурации."),
    until_stage: str | None = typer.Option(
        None, "--until-stage", help="Остановить выполнение после указанной стадии."
    ),
    force_stage: str | None = typer.Option(
        None, "--force-stage", help="Принудительно перезапустить стадию, игнорируя кеш."
    ),
) -> None:
    """Алиас для команды analyze."""
    analyze_cmd(
        vod=vod,
        chat=chat,
        game=game,
        hero=hero,
        config=config,
        until_stage=until_stage,
        force_stage=force_stage,
    )


@app.command(name="transcribe")
def transcribe_cmd(
    vod: Path = typer.Argument(..., help="Путь к файлу VOD."),
    config: Path | None = typer.Option(None, "--config", "-c", help="Путь к файлу конфигурации."),
    output_format: str = typer.Option("all", "--format", help="Формат экспорта: srt, txt, all."),
) -> None:
    """Выполнить извлечение аудио, VAD, LID и ASR-транскрипцию стрима (Этап M1)."""
    if not vod.is_file():
        console.print(f"[bold red]Ошибка:[/bold red] Файл VOD не найден: {vod}")
        raise typer.Exit(code=1)

    cfg = load_config(config_path=config)
    project = Project.create(vod_path=vod, config=cfg)

    dag = build_pipeline_dag()
    try:
        dag.run(project, until_stage="s04_asr")
        s04_dir = project.stages_dir / "s04_asr"
        console.print("\n[bold green][OK] Транскрипция успешно завершена![/bold green]")
        console.print(f"Субтитры SRT: [cyan]{s04_dir / 'transcript.srt'}[/cyan]")
        console.print(f"Транскрипт TXT: [cyan]{s04_dir / 'transcript.txt'}[/cyan]")
        console.print(f"Полные данные JSON: [cyan]{s04_dir / 'transcript.json'}[/cyan]")
    except Exception as exc:
        console.print(f"[bold red]Ошибка при транскрипции:[/bold red] {exc}")
        raise typer.Exit(code=3) from exc

    raise typer.Exit(code=0)


@eval_app.command(name="asr")
def eval_asr_cmd(
    gold: Path = typer.Option(..., "--gold", help="Папка с эталонными транскриптами (eval/asr_gold)."),
    backends: str | None = typer.Option(
        None, "--backends", help="Список бэкендов через запятую (gigaam, faster_whisper)."
    ),
    config: Path | None = typer.Option(None, "--config", "-c", help="Путь к файлу конфигурации."),
) -> None:
    """Рассчитать WER и CER на эталонных аудиофрагментах (Этап M1)."""
    if not gold.is_dir():
        console.print(f"[bold red]Ошибка:[/bold red] Папка эталонов не найдена: {gold}")
        raise typer.Exit(code=1)

    cfg = load_config(config_path=config)
    backend_list = [b.strip() for b in backends.split(",")] if backends else ["faster_whisper", "gigaam"]

    gold_txt_files = list(gold.glob("*.txt"))
    if not gold_txt_files:
        console.print(f"[bold red]Ошибка:[/bold red] В папке {gold} не найдено эталонных файлов .txt")
        raise typer.Exit(code=1)

    console.print(f"[bold cyan]Запуск оценки ASR на {len(gold_txt_files)} эталонах...[/bold cyan]")

    for b_name in backend_list:
        try:
            backend = get_asr_backend(b_name)
        except Exception as exc:
            console.print(f"[bold red]Пропуск бэкенда {b_name}:[/bold red] {exc}")
            continue

        predictions: dict[str, str] = {}
        for g_file in gold_txt_files:
            stem = g_file.stem
            audio_candidate = gold / f"{stem}.wav"
            if not audio_candidate.is_file():
                console.print(f"[yellow]Предупреждение: Аудиофайл {audio_candidate.name} не найден, пропуск[/yellow]")
                continue

            info = sf.info(str(audio_candidate))
            duration_s = info.duration

            # Create full segment for this sample
            seg = SpeechSegment(
                id=0,
                start=0.0,
                end=round(duration_s, 3),
                lang="ru",
                lang_confidence=1.0,
                speaker="streamer",
                source_track="mic",
            )

            try:
                res_segs = backend.transcribe(
                    audio_path=audio_candidate,
                    segments=[seg],
                    config=cfg.asr,
                    device=cfg.device,
                )
                text = " ".join(s.text for s in res_segs).strip()
                predictions[stem] = text
            except Exception as exc:
                console.print(f"[red]Ошибка инференса {b_name} на {audio_candidate.name}:[/red] {exc}")
                predictions[stem] = ""

        if predictions:
            results = evaluate_gold_directory(gold_dir=gold, predictions=predictions)
            print_evaluation_report(results, backend_name=b_name)

    raise typer.Exit(code=0)


@app.command(name="render")
def render_cmd(
    project: Path = typer.Argument(..., help="Путь к папке проекта (projects/<name>/)."),
    top: int = typer.Option(20, "--top", help="Количество лучших моментов для рендера."),
    fast: bool = typer.Option(False, "--fast", help="Быстрый рендер по ключевым кадрам (-c copy)."),
    ids: str | None = typer.Option(None, "--ids", help="Список ID моментов через запятую (например, m_0001,m_0003)."),
) -> None:
    """Выполнить аппаратный NVENC-рендер клипов (Этап M3)."""
    if not project.is_dir():
        console.print(f"[bold red]Ошибка:[/bold red] Папка проекта не найдена: {project}")
        raise typer.Exit(code=1)

    moments_file = project / "moments.json"
    if not moments_file.is_file():
        moments_file = project / "stages" / "s12_clips" / "moments.json"
    if not moments_file.is_file():
        console.print(f"[bold red]Ошибка:[/bold red] moments.json не найден в {project}")
        raise typer.Exit(code=1)

    with open(moments_file, encoding="utf-8") as f:
        doc = MomentsDocument.model_validate_json(f.read())

    vod_path = Path(doc.vod.path)
    if not vod_path.is_file():
        console.print(f"[bold red]Ошибка:[/bold red] Исходный VOD файл не найден: {vod_path}")
        raise typer.Exit(code=1)

    selected_set = None
    if ids:
        selected_set = {i.strip() for i in ids.split(",") if i.strip()}

    target_moments = doc.moments[:top]
    if selected_set:
        target_moments = [m for m in doc.moments if m.id in selected_set]

    cfg = load_config()
    clips_dir = project / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    mode_str = "Быстрый поток (-c copy)" if fast else f"NVENC {cfg.render.codec} ({cfg.render.preset}, CQ={cfg.render.cq})"
    console.print(f"[bold cyan]Запуск рендера {len(target_moments)} клипов... Режим: {mode_str}[/bold cyan]")

    updated_moments = render_clips_batch(
        vod_path=vod_path,
        moments=target_moments,
        clips_dir=clips_dir,
        render_cfg=cfg.render,
        fast=fast,
        max_workers=cfg.render.max_parallel,
    )

    # Update moments document
    m_dict = {m.id: m for m in updated_moments}
    for idx, orig_m in enumerate(doc.moments):
        if orig_m.id in m_dict:
            doc.moments[idx] = m_dict[orig_m.id]

    export_moments_json(doc, project / "moments.json")
    export_moments_csv(doc, project / "moments.csv")
    export_youtube_timestamps(doc, project / "youtube_timestamps.txt")
    out_html = render_html_report(project)

    console.print("\n[bold green][OK] Рендер клипов успешно завершен![/bold green]")
    console.print(f"Папка с клипами: [cyan]{clips_dir}[/cyan]")
    console.print(f"Обновленный отчёт: [cyan]{out_html}[/cyan]")
    raise typer.Exit(code=0)


@app.command(name="report")
def report_cmd(
    project: Path = typer.Argument(..., help="Путь к папке проекта (projects/<name>/)."),
) -> None:
    """Сгенерировать или пересобрать автономный HTML-отчёт по результатам анализа."""
    if not project.is_dir():
        console.print(f"[bold red]Ошибка:[/bold red] Папка проекта не найдена: {project}")
        raise typer.Exit(code=1)

    try:
        out_html = render_html_report(project)
        console.print(f"[bold green][OK] HTML-отчёт успешно пересобран:[/bold green] [cyan]{out_html}[/cyan]")
    except Exception as exc:
        console.print(f"[bold red]Ошибка создания отчёта:[/bold red] {exc}")
        raise typer.Exit(code=3) from exc

    raise typer.Exit(code=0)


@app.command(name="calibrate")
def calibrate_cmd(
    game: str = typer.Argument("deadlock", help="Имя игры (deadlock)."),
    vod: Path = typer.Argument(..., help="Путь к файлу VOD для калибровки."),
    at: str = typer.Option("00:10:00", "--at", help="Таймкод кадра для разметки ROI (HH:MM:SS)."),
) -> None:
    """Интерактивная калибровка областей экрана и шаблонов игры (Этап M4)."""
    console.print(
        f"[yellow]Команда 'calibrate' для {game} будет доступна на этапе M4.[/yellow]\n"
        f"VOD: {vod}, таймкод: {at}"
    )
    raise typer.Exit(code=0)


@eval_app.command(name="highlights")
def eval_highlights_cmd(
    project: Path = typer.Argument(..., help="Путь к проекту."),
    gt: Path = typer.Option(..., "--gt", help="CSV-файл с разметкой моментов пользователя."),
) -> None:
    """Рассчитать recall и precision кандидатов и отобранных моментов."""
    if not project.is_dir():
        console.print(f"[bold red]Ошибка:[/bold red] Папка проекта не найдена: {project}")
        raise typer.Exit(code=1)
    if not gt.is_file():
        console.print(f"[bold red]Ошибка:[/bold red] CSV файл разметки не найден: {gt}")
        raise typer.Exit(code=1)

    try:
        gt_items = load_ground_truth(gt)
    except Exception as exc:
        console.print(f"[bold red]Ошибка чтения эталонного файла {gt}:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    # Load candidates from s10
    cand_file = project / "stages" / "s10_candidates" / "candidates.jsonl"
    if not cand_file.is_file():
        cand_file = project / "s10_candidates" / "candidates.jsonl"

    candidates: list[Candidate] = []
    if cand_file.is_file():
        with open(cand_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    candidates.append(Candidate.model_validate_json(line))

    # Load moments from moments.json
    moments_file = project / "moments.json"
    if not moments_file.is_file():
        moments_file = project / "stages" / "s12_clips" / "moments.json"

    if not moments_file.is_file():
        console.print(f"[bold red]Ошибка:[/bold red] moments.json не найден в {project}")
        raise typer.Exit(code=1)

    with open(moments_file, encoding="utf-8") as f:
        doc = MomentsDocument.model_validate_json(f.read())

    report = evaluate_highlights(
        gt_items=gt_items,
        candidates=candidates,
        moments=doc.moments,
    )
    print_highlights_report(report, console=console)
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
