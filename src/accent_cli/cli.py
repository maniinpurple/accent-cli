"""accent-cli — record or load speech, extract prosody metrics, hand them to an AI coach."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from . import analyzer, prompts

# .env in the current directory first, then the project root (editable install).
load_dotenv(Path.cwd() / ".env")
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

app = typer.Typer(
    add_completion=False,
    help="Personal accent coach: record → measure rhythm & prosody → paste into an AI coach.",
    rich_markup_mode="rich",
)
console = Console()
err = Console(stderr=True)

ModelOpt = Annotated[str, typer.Option("--model", "-m", help="faster-whisper model (base.en, small.en, ...).")]
WordsOpt = Annotated[bool, typer.Option("--words", "-w", help="Show the per-word timing/pitch table.")]
JsonOpt = Annotated[bool, typer.Option("--json", help="Print JSON to stdout and exit (no summary/menu).")]
ExportOpt = Annotated[Path, typer.Option("--export-dir", help="Where option [4] saves JSON.")]
TextOpt = Annotated[
    Optional[str],
    typer.Option("--text", "-t", help="Exact sentence you plan to say — enables the pronunciation check."),
]
RawOpt = Annotated[bool, typer.Option("--raw", help="Skip preprocessing (high-pass, silence trim, normalize).")]


# --------------------------------------------------------------------------- commands

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Run with no command to start recording."""
    if ctx.invoked_subcommand is None:
        record()


@app.command()
def record(
    model: ModelOpt = analyzer.DEFAULT_MODEL,
    words: WordsOpt = False,
    as_json: JsonOpt = False,
    export_dir: ExportOpt = Path("exports"),
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="Where to save the WAV.")] = None,
    text: TextOpt = None,
    raw: RawOpt = False,
) -> None:
    """Record from the microphone (default command)."""
    from . import recorder

    ui = err if as_json else console
    if text is None:
        text = ui.input("[bold]Target sentence[/] [dim](what you'll say — Enter to skip)[/]: ").strip() or None
    else:
        ui.print(Panel(text, title="Target sentence", border_style="cyan", expand=False))

    while True:
        try:
            wav = recorder.record_to_wav(out, console=ui, target_text=text)
        except recorder.RecorderError as exc:
            err.print(f"[red]✗ {exc}[/]")
            raise typer.Exit(1) from exc
        audio, prep, raw_words = _transcribe(wav, model, clean=not raw)
        render_heard(ui, _light_words(raw_words), text, prep)
        choice = Prompt.ask(
            "[bold]\\[c][/] Continue analysis  [bold]\\[r][/] Retake",
            choices=["c", "r"],
            default="c" if raw_words else "r",
            console=ui,
        )
        if choice == "c":
            break

    payload = _build(audio, raw_words, wav.name, model, text, prep)
    _finish(payload, show_words=words, as_json=as_json, export_dir=export_dir)


@app.command()
def analyze(
    file_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Audio file (wav/mp3/m4a…).")],
    model: ModelOpt = analyzer.DEFAULT_MODEL,
    words: WordsOpt = False,
    as_json: JsonOpt = False,
    export_dir: ExportOpt = Path("exports"),
    text: TextOpt = None,
    raw: RawOpt = False,
) -> None:
    """Analyze an existing audio file."""
    audio, prep, raw_words = _transcribe(file_path, model, clean=not raw)
    payload = _build(audio, raw_words, file_path.name, model, text, prep)
    _finish(payload, show_words=words, as_json=as_json, export_dir=export_dir)


# --------------------------------------------------------------------------- pipeline

def _transcribe(path: Path, model: str, clean: bool = True):
    """Decode + preprocess + ASR. Returns (audio, preprocessing_info, raw_words)."""
    try:
        with err.status(f"[bold]Listening back[/] to {path.name} with faster-whisper/{model}…"):
            audio, prep = analyzer.load_and_preprocess(path, clean=clean)
            raw_words = analyzer.transcribe_words(audio, model)
    except Exception as exc:  # noqa: BLE001
        err.print(f"[red]✗ Transcription failed:[/] {exc}")
        raise typer.Exit(1) from exc
    return audio, prep, raw_words


def _build(audio, raw_words, source: str, model: str, text: str | None, prep) -> dict[str, Any]:
    try:
        with err.status("[bold]Measuring[/] rhythm, pitch and formants (Praat)…"):
            return analyzer.build_payload(
                audio, raw_words, source=source, model_name=model, target_text=text, preprocessing=prep
            )
    except Exception as exc:  # noqa: BLE001
        err.print(f"[red]✗ Analysis failed:[/] {exc}")
        raise typer.Exit(1) from exc


def _light_words(raw_words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"i": k, "word": w["text"], "start_ms": int(w["start"] * 1000)} for k, w in enumerate(raw_words)]


def _finish(payload: dict[str, Any], *, show_words: bool, as_json: bool, export_dir: Path) -> None:
    if as_json:
        typer.echo(analyzer.to_json(payload, compact=False))
        return

    render_summary(payload, show_words=show_words)
    if not payload["words"]:
        console.print("[yellow]No speech detected — nothing to coach. Try recording again.[/]")
        return
    interactive_menu(payload, export_dir=export_dir)


# --------------------------------------------------------------------------- rendering

def _fmt(v: Any, suffix: str = "") -> str:
    return "—" if v is None else f"{v}{suffix}"


DIFF_LEGEND = "[dim]legend:[/] [bold red]misheard[/]  [red strike]missing[/]  [yellow]+extra[/]"


def diff_markup(target_text: str, words: list[dict[str, Any]]) -> tuple[str, str, str]:
    """(heard_line, target_line, stats_line) as Rich markup."""
    from .target import align, compare_to_target

    heard, target = [], []
    for op in align(target_text, words):
        if op.op == "equal":
            heard.append(escape(op.heard))
            target.append(f"[green]{escape(op.expected)}[/]")
        elif op.op == "substitution":
            heard.append(f"[bold red]{escape(op.heard)}[/]")
            target.append(f"[bold red underline]{escape(op.expected)}[/]")
        elif op.op == "deletion":
            target.append(f"[red strike]{escape(op.expected)}[/]")
        else:
            heard.append(f"[yellow]+{escape(op.heard)}[/]")
    cmp = compare_to_target(target_text, words)
    wer, n = cmp["word_error_rate"], len(cmp["mismatches"])
    style = "green" if wer == 0 else "yellow" if wer <= 0.15 else "red"
    stats = f"[{style}]WER {wer:.0%}[/] · {n} mismatch{'es' if n != 1 else ''}"
    return " ".join(heard) or "[dim](nothing)[/]", " ".join(target), stats


def render_heard(
    ui: Console, words: list[dict[str, Any]], target_text: str | None, prep: dict[str, Any] | None
) -> None:
    """Show what the recognizer heard right after a take, so the user can decide to retake."""
    if not words:
        body = "[yellow](nothing recognized — check the mic level, move closer, or speak louder)[/]"
    elif target_text:
        heard, target, stats = diff_markup(target_text, words)
        body = f"[dim]Heard: [/] {heard}\n[dim]Target:[/] {target}\n{stats}   {DIFF_LEGEND}"
    else:
        body = escape(" ".join(w["word"] for w in words))
    if prep:
        speech_s = (prep["original_duration_ms"] - prep["trimmed_lead_ms"] - prep["trimmed_tail_ms"]) / 1000
        body += (
            f"\n[dim]{speech_s:.1f}s of speech · trimmed {prep['trimmed_lead_ms']} ms lead / "
            f"{prep['trimmed_tail_ms']} ms tail · gain {prep['gain_db']:+.1f} dB[/]"
        )
    ui.print(Panel(body, title="What the machine heard", border_style="magenta"))


def render_summary(payload: dict[str, Any], show_words: bool = False) -> None:
    s, f = payload["summary"], payload["flags"]
    console.print()
    if payload.get("target_text") and payload["words"]:
        heard, target, stats = diff_markup(payload["target_text"], payload["words"])
        console.print(Panel(f"[dim]Heard: [/] {heard}\n[dim]Target:[/] {target}\n{stats}   {DIFF_LEGEND}",
                            title="Transcript vs target", border_style="cyan"))
    else:
        console.print(Panel(escape(payload["transcript"]) or "[dim](no speech)[/]", title="Transcript", border_style="cyan"))

    rhythm_style = {"stress-timed": "green", "mixed": "yellow", "syllable-timed": "red"}.get(
        s.get("rhythm_class", ""), "white"
    )
    t = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2))
    t.add_column(style="dim")
    t.add_column()
    t.add_row("nPVI (rhythm)", f"[bold {rhythm_style}]{_fmt(s.get('npvi'))}[/]  [{rhythm_style}]{s.get('rhythm_class')}[/]")
    t.add_row("Function / content dur", f"{_fmt(s.get('function_content_ratio'))}  "
              f"[dim]({_fmt(s.get('function_word_avg_ms'), 'ms')} vs {_fmt(s.get('content_word_avg_ms'), 'ms')}, target ≤ 0.55)[/]")
    t.add_row("Articulation rate", f"{_fmt(s.get('articulation_rate_sps'))} syll/s  [dim](words: {s.get('word_count')})[/]")
    t.add_row("Pitch", f"{_fmt(s.get('pitch_mean_hz'), ' Hz')} mean, range {_fmt(s.get('pitch_range_st'), ' st')}")
    if payload.get("word_error_rate") is not None:
        t.add_row("Pronunciation (WER)", f"{payload['word_error_rate']:.0%}  "
                  f"[dim]({len(payload['mismatches'])} mismatches vs target)[/]")
    t.add_row("Pauses ≥250ms", f"{s.get('pause_count', 0)}  [dim]({s.get('pause_total_ms', 0)} ms total)[/]")
    console.print(Panel(t, title="Metrics", border_style="cyan"))

    lines = []
    if f.get("slow_function_words"):
        lines.append("[bold]Slow function words:[/] " + ", ".join(
            f"[red]{w['word']}[/] {w['duration_ms']}ms" for w in f["slow_function_words"]))
    if f.get("long_pauses"):
        lines.append("[bold]Pauses:[/] " + ", ".join(
            f"after [yellow]{p['after_word']}[/] {p['pause_ms']}ms" + ("" if p["at_punctuation"] else " [red](mid-phrase)[/]")
            for p in f["long_pauses"]))
    if f.get("low_confidence_words"):
        lines.append("[bold]Unclear words:[/] " + ", ".join(
            f"[magenta]{w['word']}[/] ({w['conf']})" for w in f["low_confidence_words"]))
    if f.get("flap_candidates"):
        lines.append("[bold]Flap/link spots:[/] " + ", ".join(f"“{c['text']}”" for c in f["flap_candidates"]))
    if lines:
        console.print(Panel("\n".join(lines), title="Flags", border_style="yellow"))

    if show_words and payload["words"]:
        wt = Table(box=None, header_style="bold dim")
        for col in ("#", "word", "start", "dur", "pause", "F0 avg", "F0 std", "drop", "dB", "conf"):
            wt.add_column(col, justify="left" if col == "word" else "right")
        for w in payload["words"]:
            style = "cyan" if w["function_word"] else ""
            wt.add_row(str(w["i"]), f"[{style}]{w['word']}[/]" if style else w["word"], str(w["start_ms"]),
                       str(w["duration_ms"]), _fmt(w["pause_after_ms"]), _fmt(w["pitch_avg_hz"]),
                       _fmt(w["pitch_std_hz"]), _fmt(w["pitch_end_drop_hz"]), _fmt(w["intensity_db"]), _fmt(w["conf"]))
        console.print(Panel(wt, title="Words [dim](cyan = function word, ms / Hz)[/]", border_style="dim"))


# --------------------------------------------------------------------------- menu

MENU = """\
[bold][1][/] Copy Prompt + JSON [dim](ready to paste into AI chat)[/]
[bold][2][/] Copy AI instruction prompt only
[bold][3][/] Copy JSON data only
[bold][4][/] Save JSON to {export_dir}/accent_<timestamp>.json
[bold][5][/] Run direct AI analysis via API (Gemini/Claude)
[bold]\\[q][/] Quit"""


def copy_to_clipboard(text: str, label: str) -> bool:
    import pyperclip

    try:
        pyperclip.copy(text)
    except pyperclip.PyperclipException as exc:
        console.print(f"[red]✗ Clipboard unavailable ({exc}). Use [4] to save instead.[/]")
        return False
    console.print(f"[green]✓ Copied {label}[/] [dim]({len(text):,} chars)[/]")
    return True


def interactive_menu(payload: dict[str, Any], export_dir: Path = Path("exports")) -> None:
    payload_json = analyzer.to_json(payload, compact=True)
    while True:
        console.print(Panel(MENU.format(export_dir=export_dir), title="Next", border_style="green", expand=False))
        choice = Prompt.ask("Choose", choices=["1", "2", "3", "4", "5", "q"], default="1", console=console)
        if choice == "1":
            copy_to_clipboard(prompts.build_combined(payload_json), "prompt + JSON")
        elif choice == "2":
            copy_to_clipboard(prompts.AI_COACH_PROMPT, "AI coach prompt")
        elif choice == "3":
            copy_to_clipboard(payload_json, "JSON")
        elif choice == "4":
            path = analyzer.save_json(payload, export_dir)
            console.print(f"[green]✓ Saved[/] {path}")
        elif choice == "5":
            run_api_analysis(payload_json, export_dir)
        else:
            return


def run_api_analysis(payload_json: str, export_dir: Path) -> None:
    from .providers import ProviderError, configured_providers, get_provider

    available = configured_providers()
    if not available:
        console.print("[yellow]No API key found.[/] Copy [bold].env.example[/] → [bold].env[/] and set "
                      "GEMINI_API_KEY and/or ANTHROPIC_API_KEY.")
        return
    name = available[0] if len(available) == 1 else Prompt.ask(
        "Provider", choices=available, default=available[0], console=console)

    try:
        provider = get_provider(name)
        text = ""
        console.print(f"[dim]→ {provider.name} / {provider.model}[/]")
        with Live(Markdown(""), console=console, refresh_per_second=8, vertical_overflow="visible") as live:
            for chunk in provider.stream(prompts.AI_COACH_PROMPT, prompts.build_user_message(payload_json)):
                text += chunk
                live.update(Markdown(text))
    except ProviderError as exc:
        console.print(f"[red]✗ {exc}[/]")
        return

    if text.strip():
        export_dir.mkdir(parents=True, exist_ok=True)
        path = export_dir / f"accent_{datetime.now():%Y%m%d_%H%M%S}_{name}.md"
        path.write_text(text, encoding="utf-8")
        console.print(f"[dim]Saved feedback → {path}[/]")


if __name__ == "__main__":  # pragma: no cover
    app()
