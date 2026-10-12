"""Environment diagnostics module for verifying Python, FFmpeg, CUDA, NVENC, and sm_120 compatibility."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from narezka.config import AppConfig

# Ensure Windows console uses UTF-8 if supported
if sys.platform == "win32":
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

console = Console(force_terminal=True)


class DiagnosticItem(BaseModel):
    category: str
    name: str
    status: str  # "PASS", "WARN", "FAIL"
    details: str
    recommendation: str | None = None


def check_python() -> DiagnosticItem:
    """Check Python version (must be >= 3.12)."""
    major, minor, micro = sys.version_info[:3]
    version_str = f"{major}.{minor}.{micro}"
    if major == 3 and minor >= 12:
        return DiagnosticItem(
            category="Окружение",
            name="Версия Python",
            status="PASS",
            details=f"Python {version_str} ({sys.executable})",
        )
    return DiagnosticItem(
        category="Окружение",
        name="Версия Python",
        status="FAIL",
        details=f"Python {version_str} (требуется >= 3.12)",
        recommendation="Установите Python 3.12+ через официальный установщик или 'uv python install 3.12'.",
    )


def check_ffmpeg() -> DiagnosticItem:
    """Verify presence of ffmpeg and ffprobe executables in PATH."""
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    if not ffmpeg_path or not ffprobe_path:
        missing = []
        if not ffmpeg_path:
            missing.append("ffmpeg")
        if not ffprobe_path:
            missing.append("ffprobe")
        return DiagnosticItem(
            category="Медиа",
            name="FFmpeg / FFprobe",
            status="FAIL",
            details=f"Не найдены в PATH: {', '.join(missing)}",
            recommendation="Установите полный билд FFmpeg (gyan.dev) и добавьте папку bin в системный PATH.",
        )

    try:
        res = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True,
        )
        first_line = res.stdout.splitlines()[0] if res.stdout else "FFmpeg найден"
        return DiagnosticItem(
            category="Медиа",
            name="FFmpeg / FFprobe",
            status="PASS",
            details=first_line,
        )
    except Exception as exc:
        return DiagnosticItem(
            category="Медиа",
            name="FFmpeg / FFprobe",
            status="FAIL",
            details=f"Ошибка при запуске ffmpeg: {exc}",
            recommendation="Проверьте целостность установленного FFmpeg.",
        )


def check_nvenc() -> DiagnosticItem:
    """Check availability of NVENC hardware encoders in FFmpeg and run a test encoding."""
    if not shutil.which("ffmpeg"):
        return DiagnosticItem(
            category="Медиа",
            name="NVENC аппаратное кодирование",
            status="FAIL",
            details="ffmpeg не найден",
        )

    try:
        res = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
        )
        encoders = res.stdout
        has_h264 = "h264_nvenc" in encoders
        has_hevc = "hevc_nvenc" in encoders

        if not has_h264 and not has_hevc:
            return DiagnosticItem(
                category="Медиа",
                name="NVENC аппаратное кодирование",
                status="FAIL",
                details="Энкодеры h264_nvenc / hevc_nvenc отсутствуют в ffmpeg",
                recommendation="Установите FFmpeg full build с поддержкой NVENC (например, gyan.dev/ffmpeg/builds).",
            )

        # Run test 1-second synthetic encoding to NUL
        test_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=640x360:rate=30",
            "-c:v",
            "h264_nvenc",
            "-f",
            "null",
            "NUL" if sys.platform == "win32" else "/dev/null",
        ]
        test_res = subprocess.run(
            test_cmd,
            capture_output=True,
            text=True,
        )
        if test_res.returncode != 0:
            last_err = test_res.stderr.strip().splitlines()[-3:]
            return DiagnosticItem(
                category="Медиа",
                name="NVENC аппаратное кодирование",
                status="WARN",
                details=f"Энкодер найден, но тест завершился ошибкой: {' '.join(last_err)}",
                recommendation="Проверьте актуальность драйверов NVIDIA (NVIDIA Studio / Game Ready).",
            )

        available_codecs = []
        if has_h264:
            available_codecs.append("h264_nvenc")
        if has_hevc:
            available_codecs.append("hevc_nvenc")

        return DiagnosticItem(
            category="Медиа",
            name="NVENC аппаратное кодирование",
            status="PASS",
            details=f"Доступны и протестированы: {', '.join(available_codecs)}",
        )
    except Exception as exc:
        return DiagnosticItem(
            category="Медиа",
            name="NVENC аппаратное кодирование",
            status="FAIL",
            details=f"Ошибка проверки NVENC: {exc}",
        )


def check_torch_cuda() -> DiagnosticItem:
    """Verify PyTorch installation, CUDA availability, GPU device info, and perform tensor math on sm_120."""
    try:
        import torch
    except ImportError:
        return DiagnosticItem(
            category="GPU & ML",
            name="PyTorch CUDA Runtime",
            status="FAIL",
            details="Пакет torch не установлен в текущем окружении",
            recommendation="Установите PyTorch с поддержкой CUDA 12.8: 'uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128'",
        )

    if not torch.cuda.is_available():
        return DiagnosticItem(
            category="GPU & ML",
            name="PyTorch CUDA Runtime",
            status="FAIL",
            details=f"torch {torch.__version__} установлен, но CUDA недоступна (torch.cuda.is_available() == False)",
            recommendation="Убедитесь, что установлена CUDA-версия PyTorch (например, cu128), а не CPU-колесо.",
        )

    device_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    cc_str = f"{major}.{minor}"

    # Test actual matrix multiplication on GPU to verify sm_120 PTX/SASS kernels
    try:
        t0 = time.perf_counter()
        a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        del a, b, c
        torch.cuda.empty_cache()

        is_sm120 = major >= 12
        arch_note = " (Blackwell sm_120)" if is_sm120 else ""

        return DiagnosticItem(
            category="GPU & ML",
            name="PyTorch CUDA Runtime",
            status="PASS",
            details=(
                f"GPU: {device_name} | Compute: {cc_str}{arch_note} | "
                f"CUDA Torch: {torch.version.cuda} | Тест FP16: {elapsed_ms:.1f} мс"
            ),
        )
    except Exception as exc:
        err_msg = str(exc)
        rec = "Обновите PyTorch до версии с поддержкой sm_120 (CUDA 12.8+)."
        if "no kernel image is available" in err_msg:
            rec = (
                "Критическая несовместимость с архитектурой Blackwell sm_120! "
                "Установите PyTorch с поддержкой CUDA 12.8+ из индекса: "
                "https://download.pytorch.org/whl/cu128"
            )
        return DiagnosticItem(
            category="GPU & ML",
            name="PyTorch CUDA Runtime",
            status="FAIL",
            details=f"Ошибка выполнения CUDA-операции: {err_msg}",
            recommendation=rec,
        )


def check_ctranslate2() -> DiagnosticItem:
    """Check CTranslate2 compatibility for faster-whisper on sm_120."""
    try:
        import importlib

        ctranslate2 = importlib.import_module("ctranslate2")
    except ImportError:
        return DiagnosticItem(
            category="GPU & ML",
            name="CTranslate2 (faster-whisper)",
            status="WARN",
            details="Пакет ctranslate2 не установлен (потребуется на этапе M1)",
            recommendation="Будет установлен на этапе M1 вместе с faster-whisper.",
        )

    try:
        cuda_types = ctranslate2.get_supported_compute_types("cuda")
        has_cuda = len(cuda_types) > 0
        if not has_cuda:
            return DiagnosticItem(
                category="GPU & ML",
                name="CTranslate2 (faster-whisper)",
                status="WARN",
                details="CTranslate2 установлен, но CUDA-бэкенд не поддерживается",
                recommendation="Используйте CPU-вычисления или резервный бэкенд ASR на PyTorch cu128 (HF Transformers).",
            )
        return DiagnosticItem(
            category="GPU & ML",
            name="CTranslate2 (faster-whisper)",
            status="PASS",
            details=f"Поддерживаемые compute types: {', '.join(cuda_types)}",
        )
    except Exception as exc:
        return DiagnosticItem(
            category="GPU & ML",
            name="CTranslate2 (faster-whisper)",
            status="WARN",
            details=f"Предупреждение при проверке CTranslate2: {exc}",
            recommendation="При сбое инференса на sm_120 включите резервный бэкенд Whisper через HF Transformers.",
        )


def check_onnxruntime() -> DiagnosticItem:
    """Check ONNX Runtime GPU support for audio-separator on sm_120."""
    try:
        import importlib

        ort = importlib.import_module("onnxruntime")
    except ImportError:
        return DiagnosticItem(
            category="GPU & ML",
            name="ONNX Runtime GPU",
            status="WARN",
            details="Пакет onnxruntime не установлен (потребуется на этапе M1 для audio-separator)",
            recommendation="Если onnxruntime-gpu не поддерживает sm_120, используется резервный бэкенд Demucs (PyTorch).",
        )

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return DiagnosticItem(
            category="GPU & ML",
            name="ONNX Runtime GPU",
            status="PASS",
            details=f"Провайдеры: {', '.join(providers)}",
        )
    return DiagnosticItem(
        category="GPU & ML",
        name="ONNX Runtime GPU",
        status="WARN",
        details=f"CUDAExecutionProvider отсутствует, доступны только: {', '.join(providers)}",
        recommendation="Для разделения аудио будет автоматически использован бэкенд Demucs (PyTorch cu128).",
    )


def check_disk_space(target_dir: Path | None = None) -> DiagnosticItem:
    """Check available disk space (at least 10 GB recommended for temporary files)."""
    check_path = target_dir or Path.cwd()
    try:
        usage = shutil.disk_usage(check_path)
        free_gb = usage.free / (1024**3)
        if free_gb >= 10.0:
            return DiagnosticItem(
                category="Хранилище",
                name="Свободное место на диске",
                status="PASS",
                details=f"{free_gb:.1f} GB свободно на диске ({check_path.drive or check_path})",
            )
        return DiagnosticItem(
            category="Хранилище",
            name="Свободное место на диске",
            status="WARN",
            details=f"Осталось всего {free_gb:.1f} GB на диске ({check_path.drive or check_path})",
            recommendation="Освободите минимум 10-15 GB дискового пространства для временных аудио- и видеофайлов.",
        )
    except Exception as exc:
        return DiagnosticItem(
            category="Хранилище",
            name="Свободное место на диске",
            status="WARN",
            details=f"Не удалось проверить диск: {exc}",
        )


def check_api_keys(config: AppConfig) -> DiagnosticItem:
    """Verify configured LLM Judge API keys."""
    judge_profile = config.judge.profile
    profile_cfg = config.judge.profiles.get(judge_profile)

    if not profile_cfg:
        return DiagnosticItem(
            category="Конфигурация",
            name="API ключи LLM",
            status="WARN",
            details=f"Профиль судьи '{judge_profile}' не найден в конфиге",
        )

    if judge_profile == "local":
        return DiagnosticItem(
            category="Конфигурация",
            name="API ключи LLM",
            status="PASS",
            details=f"Выбран локальный профиль ({profile_cfg.base_url}, модель: {profile_cfg.model}) — API-ключ не требуется",
        )

    api_key_env = profile_cfg.api_key_env
    if api_key_env and os.getenv(api_key_env):
        val = os.getenv(api_key_env, "")
        masked = val[:4] + "..." + val[-3:] if len(val) > 7 else "***"
        return DiagnosticItem(
            category="Конфигурация",
            name="API ключи LLM",
            status="PASS",
            details=f"Переменная окружения '{api_key_env}' установлена ({masked})",
        )

    return DiagnosticItem(
        category="Конфигурация",
        name="API ключи LLM",
        status="WARN",
        details=f"Переменная окружения '{api_key_env}' для профиля '{judge_profile}' не найдена",
        recommendation=f"Укажите ключ в файле .env: {api_key_env}=your_api_key_here",
    )


def run_doctor(config: AppConfig | None = None) -> bool:
    """Run all diagnostics and render results using Rich."""
    active_config = config or AppConfig()

    console.print(
        Panel.fit(
            "[bold cyan]Диагностика окружения «Нарезка» (narezka doctor)[/bold cyan]\n"
            "Проверка готовности системы, видеокарты RTX 5070 (Blackwell), FFmpeg и зависимостей",
            border_style="cyan",
        )
    )

    checks: list[DiagnosticItem] = [
        check_python(),
        check_ffmpeg(),
        check_nvenc(),
        check_torch_cuda(),
        check_ctranslate2(),
        check_onnxruntime(),
        check_disk_space(),
        check_api_keys(active_config),
    ]

    table = Table(title="Результаты диагностики", show_lines=True)
    table.add_column("Категория", style="cyan", no_wrap=True)
    table.add_column("Компонент", style="bold")
    table.add_column("Статус", justify="center")
    table.add_column("Детали", style="dim")
    table.add_column("Рекомендации / Пути обхода", style="yellow")

    all_passed = True
    has_critical_failure = False

    for item in checks:
        if item.status == "PASS":
            status_text = "[bold green]PASS[/bold green]"
        elif item.status == "WARN":
            status_text = "[bold yellow]WARN[/bold yellow]"
            all_passed = False
        else:
            status_text = "[bold red]FAIL[/bold red]"
            all_passed = False
            has_critical_failure = True

        table.add_row(
            item.category,
            item.name,
            status_text,
            item.details,
            item.recommendation or "-",
        )

    console.print(table)

    if has_critical_failure:
        console.print(
            "\n[bold red][FAIL] Обнаружены критические ошибки окружения. "
            "Исправьте их перед переходом к следующим этапам.[/bold red]"
        )
        return False

    if not all_passed:
        console.print(
            "\n[bold yellow][WARN] Обнаружены предупреждения. "
            "Система готова к этапу M0, но для M1 потребуется доустановить отмеченные ML-пакеты.[/bold yellow]"
        )
        return True

    console.print("\n[bold green][OK] Все проверки успешно пройдены! Система полностью готова.[/bold green]")
    return True
