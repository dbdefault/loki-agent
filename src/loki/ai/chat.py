import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any
from collections.abc import Callable
import typer

from loki import __version__
from loki.ai import llm
from loki.config import (
    activate_model_profile,
    add_model_profile,
    clear_active_model_profile,
    get_active_model_profile,
    list_model_profiles,
    model_source,
    remove_model_profile,
    resolve_model,
)
from loki.engine.updater import check_for_updates, print_update_banner, perform_update

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.status import Status

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import Completion, NestedCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Float, FloatContainer, HSplit
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea


CLI_SLASH_COMMANDS = ("run", "fix", "report")


def _cli_group():
    """The `loki` CLI as a Typer/Click command group (imported lazily: loki.cli imports this module)."""
    from loki.cli import app
    return typer.main.get_command(app)


def _cli_long_flags(command: str) -> dict[str, None]:
    """Long option names of a `loki` CLI command, for slash-command autocomplete."""
    params = _cli_group().commands[command].params
    flags = [
        opt
        for param in params
        if param.param_type_name == "option"
        for opt in [*param.opts, *param.secondary_opts]
        if opt.startswith("--")
    ]
    return dict.fromkeys(flags)


def _split_slash_args(args: str) -> list[str]:
    """Shell-like split that honors quotes but keeps backslashes literal (Windows paths)."""
    lexer = shlex.shlex(args, posix=True)
    lexer.whitespace_split = True
    lexer.escape = ""
    return list(lexer)


class _SlashNestedCompleter(NestedCompleter):
    """NestedCompleter's top-level fallback (no space typed yet) uses WordCompleter,
    whose default word-boundary rules don't treat '/' as part of a word — so typing
    '/mo' is matched as just 'mo', which never prefixes '/model'. This overrides only
    that fallback with a plain, literal prefix match; nested dispatch (once there's a
    space, e.g. '/model rem') is unaffected and still delegates to the normal logic."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        if " " in text:
            yield from super().get_completions(document, complete_event)
            return
        for key in self.options:
            if key.lower().startswith(text.lower()):
                yield Completion(key, start_position=-len(text))


class LokiChatSession:
    """Interactive conversational terminal REPL for pair QA testing, incident queries, and advice."""

    def __init__(self, model: str | None = None, runs_dir: str = ".loki/runs"):
        self._explicit_model = model
        self.model = resolve_model(model)
        self.runs_dir = Path(runs_dir)
        self.console = Console()
        self.history: list[dict[str, str]] = []
        self._input_history = InMemoryHistory()
        self._use_boxed_prompt = True

    def _load_project_context(self) -> str:
        """Assembles repository context: config, rules, and knowledge."""
        context_parts = []

        # 1. Configuration
        config_file = Path(".loki/config.yaml")
        if config_file.exists():
            context_parts.append(f"### Project Configuration (.loki/config.yaml):\n{config_file.read_text(encoding='utf-8')}")

        # 2. Business Rules
        rules_file = Path(".loki/rules.md")
        if rules_file.exists():
            context_parts.append(f"### Active Business Rules (.loki/rules.md):\n{rules_file.read_text(encoding='utf-8')}")

        # 3. Knowledge / Stack
        knowledge_file = Path(".loki/knowledge.json")
        if knowledge_file.exists():
            context_parts.append(f"### Tech Stack Knowledge (.loki/knowledge.json):\n{knowledge_file.read_text(encoding='utf-8')}")

        # 4. Recent Runs Summary
        recent_runs = self._get_recent_runs_summary(limit=3)
        if recent_runs:
            context_parts.append(f"### Recent Test & Incident Runs:\n{recent_runs}")

        return "\n\n".join(context_parts)

    def _get_recent_runs_summary(self, limit: int = 3) -> str:
        """Summarizes recent test runs and incidents."""
        if not self.runs_dir.exists():
            return "No previous runs recorded."

        run_dirs = [d for d in self.runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        if not run_dirs:
            return "No previous runs recorded."

        run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        summaries = []

        for d in run_dirs[:limit]:
            meta_file = d / "incident.json"
            if meta_file.exists():
                try:
                    with open(meta_file, encoding="utf-8") as f:
                        data = json.load(f)
                    crashes_cnt = data.get("unique_crashes_count", 0)
                    rules_cnt = len(data.get("rules_evaluations", []))
                    summaries.append(
                        f"- Run `{d.name}` ({data.get('timestamp', 'unknown')}): "
                        f"Persona: {data.get('persona', 'unknown')}, "
                        f"Target: {data.get('target_url')}, "
                        f"Crashes: {crashes_cnt}, "
                        f"Rules Evaluated: {rules_cnt}"
                    )
                except Exception:
                    pass

        return "\n".join(summaries) if summaries else "No readable runs found."

    def _display_runs_table(self):
        """Displays a Rich table of recorded runs."""
        if not self.runs_dir.exists():
            self.console.print("[yellow]No runs directory found.[/yellow]")
            return

        run_dirs = [d for d in self.runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        if not run_dirs:
            self.console.print("[yellow]No recorded runs yet.[/yellow]")
            return

        run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        table = Table(title="📦 LOKI Recorded Test Runs", border_style="cyan")
        table.add_column("Run ID", style="bold cyan")
        table.add_column("Persona", style="magenta")
        table.add_column("Crashes", justify="center")
        table.add_column("Verdict", justify="center")
        table.add_column("HTML Report", style="dim")

        for d in run_dirs[:10]:
            meta_file = d / "incident.json"
            if meta_file.exists():
                try:
                    with open(meta_file, encoding="utf-8") as f:
                        data = json.load(f)
                    crashes = data.get("unique_crashes_count", 0)
                    rules = data.get("rules_evaluations", [])
                    has_violation = any(r.get("status") == "VIOLATED" for r in rules)
                    verdict = "[red]FAIL[/red]" if (crashes > 0 or has_violation) else "[green]PASS[/green]"
                    report_path = f"{d.name}/report.html"
                    table.add_row(
                        d.name,
                        data.get("persona", "Unknown"),
                        str(crashes),
                        verdict,
                        report_path,
                    )
                except Exception:
                    continue

        self.console.print(table)

    def _display_rules(self):
        """Displays active business rules from .loki/rules.md."""
        rules_file = Path(".loki/rules.md")
        if not rules_file.exists():
            self.console.print("[yellow]No .loki/rules.md file found.[/yellow]")
            return
        self.console.print(Panel(rules_file.read_text(encoding="utf-8"), title="📋 Active Business Rules", border_style="cyan"))

    def _show_model_status(self):
        """Displays the current effective model, where it came from, and saved profiles."""
        current = resolve_model(self._explicit_model)
        source = model_source(self._explicit_model)
        profiles = list_model_profiles()
        active = get_active_model_profile()
        active_name = active["name"] if active else None

        lines = [f"[bold white]Current model:[/bold white] [cyan]{current}[/cyan] [dim]({source})[/dim]", ""]
        if profiles:
            lines.append("[bold white]Saved profiles:[/bold white]")
            for p in profiles:
                marker = "[bold green]▸[/bold green]" if p.get("name") == active_name else " "
                extra = "".join(
                    f" [dim]{k}={v}[/dim]" for k, v in (("api_base", p.get("api_base")), ("api_key_env", p.get("api_key_env"))) if v
                )
                lines.append(f"  {marker} [bold]{p['name']}[/bold] → {p['model']}{extra}")
        else:
            lines.append("[dim]No saved profiles yet — every LOKI AI feature (chat, rules, fix) is reading straight from .loki/config.yaml.[/dim]")

        lines.extend([
            "",
            "[bold white]Commands:[/bold white]",
            "  [cyan]/model <name>[/cyan]                     — switch to a saved profile",
            "  [cyan]/model add <name> <model-id> [api_base=...] [api_key_env=...][/cyan] — save & switch",
            "  [cyan]/model remove <name>[/cyan]              — delete a saved profile",
            "  [cyan]/model reset[/cyan]                       — revert to .loki/config.yaml's default",
            "",
            "[dim]e.g. /model add local ollama/llama3[/dim]",
            "[dim]e.g. /model add work mistral/mistral-large-latest api_key_env=MISTRAL_API_KEY[/dim]",
        ])
        self.console.print(Panel("\n".join(lines), title="🧠 AI Model", border_style="cyan"))

    def _handle_model_command(self, args: str):
        """Parses and executes a `/model ...` command, switching every LOKI AI
        feature (chat, rules evaluation, fix, auto-heal) to the chosen connection."""
        parts = args.split()
        if not parts:
            self._show_model_status()
            return

        sub = parts[0].lower()
        reserved = {"add", "remove", "reset", "default"}

        if sub == "add":
            if len(parts) < 3:
                self.console.print("[yellow]Usage: /model add <name> <model-id> [api_key=<KEY>] [api_base=<url>] [api_key_env=<VAR>][/yellow]")
                return
            name, model_id = parts[1], parts[2]
            if name.lower() in reserved:
                self.console.print(
                    f"[yellow]'{name}' is a reserved /model command name (add/remove/reset/default) and can't be "
                    f"used as a profile name — you'd never be able to switch back to it by name. Pick another.[/yellow]"
                )
                return
            api_base, api_key_env, api_key = None, None, None
            for token in parts[3:]:
                if token.startswith("api_base="):
                    api_base = token.split("=", 1)[1]
                elif token.startswith("api_key_env="):
                    api_key_env = token.split("=", 1)[1]
                elif token.startswith("api_key="):
                    api_key = token.split("=", 1)[1]
            add_model_profile(name, model_id, api_base=api_base, api_key_env=api_key_env, api_key=api_key)
            activate_model_profile(name)
            self._explicit_model = None
            self.model = resolve_model(None)
            self.console.print(f"[bold green]✔ Saved and switched to profile '{name}' → {model_id}[/bold green]")
            return

        if sub == "remove":
            if len(parts) < 2:
                self.console.print("[yellow]Usage: /model remove <name>[/yellow]")
                return
            name = parts[1]
            if remove_model_profile(name):
                self._explicit_model = None
                self.model = resolve_model(None)
                self.console.print(f"[green]✔ Removed profile '{name}'. Now using: {self.model}[/green]")
            else:
                self.console.print(f"[yellow]No profile named '{name}' found.[/yellow]")
            return

        if sub in ("reset", "default"):
            clear_active_model_profile()
            self._explicit_model = None
            self.model = resolve_model(None)
            self.console.print(f"[green]✔ Reverted to .loki/config.yaml's default: {self.model}[/green]")
            return

        # Otherwise: treat the argument as a profile name to switch to
        name = parts[0]
        profile = activate_model_profile(name)
        if not profile and "/" in name:
            # Looks like a bare "provider/model" id rather than a saved profile name
            # — add it under its own name so it's there next time too.
            add_model_profile(name, name)
            profile = activate_model_profile(name)
        if not profile:
            self.console.print(
                f"[yellow]No saved profile named '{name}'.[/yellow] "
                f"Use [bold]/model add {name} <model-id>[/bold] to create it, or [bold]/model[/bold] to see what's saved."
            )
            return

        self._explicit_model = None
        self.model = resolve_model(None)
        self.console.print(f"[bold green]✔ Switched active model to '{profile['name']}' → {profile['model']}[/bold green]")

    def _run_cli_action(self, description: str, fn: Callable, **kwargs):
        """Invokes a `loki` CLI command function directly — bypassing Click/Typer's own
        argument parsing — so /run, /fix, and /report drive the exact same code path as
        the standalone CLI commands, instead of a separate reimplementation. A command
        function's own `typer.Exit(code=...)` or `SystemExit(code=...)` is treated as a
        normal, expected ending (it's how e.g. the --ci gate signals pass/fail), not an
        error; anything else unexpected is reported without killing the chat session."""
        try:
            fn(**kwargs)
        except (typer.Exit, SystemExit) as e:
            code = getattr(e, "exit_code", None)
            msg = None
            if code is None:
                raw = getattr(e, "code", None)
                if raw is None:
                    code = 0
                elif isinstance(raw, int):
                    code = raw
                else:
                    code = 1
                    msg = str(raw)
            if code not in (0, None):
                suffix = f": {msg}" if msg else ""
                self.console.print(f"[dim]({description} finished with exit code {code}{suffix})[/dim]")
        except Exception as e:
            self.console.print(f"[bold red]Error running {description}:[/bold red] {e}")

    def _invoke_cli_command(self, command: str, args: str):
        """Runs `loki <command> <args>` for /run, /fix and /report through the CLI's own
        Typer parser, so the slash commands accept exactly the same flags, defaults,
        validation and `--help` as the standalone commands — no second parser to keep in sync."""
        try:
            tokens = _split_slash_args(args)
        except ValueError as e:
            self.console.print(f"[yellow]Could not parse /{command} arguments: {e}[/yellow]")
            return

        def invoke():
            try:
                exit_code = _cli_group().main(args=[command, *tokens], prog_name="loki", standalone_mode=False)
            except typer.TyperException as e:
                message = e.format_message() if hasattr(e, "format_message") else str(e)
                self.console.print(f"[yellow]{message}[/yellow] [dim](see /{command} --help)[/dim]")
                return
            if exit_code:
                raise typer.Exit(code=exit_code)

        self._run_cli_action(f"loki {command}", invoke)

    def _handle_update_command(self):
        """Checks for updates and executes the self-update via uv if requested."""
        with Status("[bold yellow]Checking for updates on GitHub...[/bold yellow]", console=self.console):
            update_info = check_for_updates(force=True)

        if not update_info:
            self.console.print("[yellow]Could not reach GitHub. Check your internet connection.[/yellow]")
            return

        if not update_info.get("available"):
            curr = update_info.get("current_version", __version__)
            self.console.print(f"[bold green]✔ You are already on the latest version of LOKI (v{curr}).[/bold green]")
            return

        print_update_banner(self.console, update_info)

        from rich.prompt import Confirm
        try:
            if Confirm.ask("Do you want to update LOKI now via uv?", default=True):
                updated = perform_update(self.console)
                if updated:
                    self.console.print()
                    if Confirm.ask("Exit interactive chat now to load the updated version?", default=True):
                        self.console.print("[dim]Exiting LOKI. Run 'loki chat' to start using the updated version! 🛡️[/dim]")
                        sys.exit(0)
                    else:
                        self.console.print("[yellow]Continuing session with current in-memory version. Please restart LOKI when finished.[/yellow]")
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[dim]Update cancelled.[/dim]")

    def _build_completer(self) -> NestedCompleter:
        """Builds a fresh Tab/as-you-type completer, including saved /model profile names."""
        profile_names = [p["name"] for p in list_model_profiles()]
        model_targets: dict[str, Any] = {name: None for name in profile_names}
        model_targets.update({
            "add": None,
            "remove": {name: None for name in profile_names} if profile_names else None,
            "reset": None,
            "default": None,
        })
        return _SlashNestedCompleter.from_nested_dict({
            "/model": model_targets,
            **{f"/{c}": _cli_long_flags(c) for c in CLI_SLASH_COMMANDS},
            "/runs": None,
            "/rules": None,
            "/update": None,
            "/help": None,
            "/clear": None,
            "exit": None,
            "quit": None,
        })

    _BOX_STYLE = Style.from_dict({"frame.border": "fg:#f85149"})

    def _read_boxed_input(self) -> str:
        """Renders a genuine bordered text box (all four sides, live, while typing) with
        command autocomplete, using a real prompt_toolkit widget — not hand-drawn ASCII
        lines printed before/after, which only ever show a floating top edge while typing."""

        def accept(buff):
            get_app().exit(result=buff.text)
            return True

        text_area = TextArea(
            multiline=False,
            wrap_lines=False,
            completer=self._build_completer(),
            complete_while_typing=True,
            history=self._input_history,
            accept_handler=accept,
        )

        kb = KeyBindings()

        @kb.add("c-c")
        def _(event):
            raise KeyboardInterrupt()

        @kb.add("c-d")
        def _(event):
            raise EOFError()

        # FloatContainer + CompletionsMenu is what actually draws the suggestions
        # dropdown near the cursor as you type; without it, completion still works
        # internally (Tab/typing filters candidates) but nothing is ever shown.
        root_container = FloatContainer(
            content=HSplit([Frame(text_area)]),
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(max_height=8, scroll_offset=1),
                ),
            ],
        )

        app = Application(
            layout=Layout(root_container, focused_element=text_area),
            key_bindings=kb,
            style=self._BOX_STYLE,
            full_screen=False,
        )
        return app.run() or ""

    def _read_input(self) -> str:
        """Reads one line of user input from a boxed prompt with live command autocomplete.
        Falls back to a plain prompt for the rest of the session if the terminal can't
        support prompt_toolkit's rendering (e.g. some non-native consoles, piped/
        non-interactive input, or CI)."""
        if self._use_boxed_prompt and not (sys.stdin.isatty() and sys.stdout.isatty()):
            # No real interactive terminal (piped input, CI, some redirected subprocess
            # setups) — skip straight to plain input. Attempting prompt_toolkit here risks
            # it consuming/losing the first line of input before it fails.
            self._use_boxed_prompt = False

        if self._use_boxed_prompt:
            try:
                return self._read_boxed_input().strip().lstrip("﻿")
            except (KeyboardInterrupt, EOFError):
                raise
            except Exception as e:
                # Terminal doesn't support prompt_toolkit's rendering — fall back below,
                # for this and every later turn this session. Note it once rather than
                # failing silently, so a genuine bug here doesn't just look like "the
                # box feature quietly isn't available."
                self.console.print(f"[dim yellow](boxed input unavailable: {e}; switching to plain prompt)[/dim yellow]")
                self._use_boxed_prompt = False

        return self.console.input("\n[bold red]❯[/bold red] ").strip().lstrip("﻿")

    def start(self):
        """Launches the interactive REPL chat session."""
        self.console.print(
            Panel(
                fr"""[bold red]  _       ____  _  __ _____     _____ _           _   [/bold red]
[bold red] | |     / __ \| |/ /|_   _|   / ____| |         | |  [/bold red]
[bold red] | |    | |  | | ' /   | |    | |    | |__   __ _| |_ [/bold red]
[bold red] | |    | |  | |  <    | |    | |    | '_ \ / _` | __|[/bold red]
[bold red] | |____| |__| | . \  _| |_   | |____| | | | (_| | |_ [/bold red]
[bold red] |______|\____/|_|\_\|_____|   \_____|_| |_|\__,_|\__|[/bold red]

[bold white]Interactive AI QA & Chaos Testing Assistant[/bold white]
[dim]Powered by LiteLLM ({self.model})[/dim]

[cyan]Commands:[/cyan] [bold]/help[/bold] (commands), [bold]/model[/bold] (switch AI model), [bold]/runs[/bold] (list runs), [bold]/rules[/bold] (view rules), [bold]/update[/bold] (upgrade LOKI), [bold]/clear[/bold] (clear screen), [bold]exit[/bold] (quit)""",
                border_style="red",
            )
        )

        # Check for available updates on GitHub and notify user
        try:
            update_info = check_for_updates()
            if update_info and update_info.get("available"):
                print_update_banner(self.console, update_info)
        except Exception:
            pass

        project_context = self._load_project_context()

        system_instruction = f"""You are LOKI, an elite AI Chaos & Software Quality Assurance Engineer.
You specialize in synthetic user simulation, race conditions, edge-case vulnerability testing, and automated root-cause analysis.

You are interacting live with the software developer or QA engineer via an interactive terminal REPL.
Answer questions directly, accurately, and concisely. When explaining crashes or suggesting fixes, provide exact code blocks or actionable guidance.

Project & Testing Context:
{project_context}
"""

        self.history.append({"role": "system", "content": system_instruction})

        while True:
            self.console.print()
            try:
                user_input = self._read_input()
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[dim]Session terminated. Goodbye![/dim]")
                break

            if not user_input:
                continue

            # Command dispatch
            cmd_lower = user_input.lower()
            if cmd_lower in ["exit", "quit", ":q"]:
                self.console.print("[dim]Exiting LOKI chat. Happy testing![/dim]")
                break

            if cmd_lower == "/help":
                self.console.print(
                    Panel(
                        "• [bold cyan]/run [url] [flags][/bold cyan] — Launch a real chaos attack (same flags as `loki run`), without leaving chat\n"
                        "• [bold cyan]/fix [run_id] [--apply][/bold cyan] — Diagnose (and optionally patch) the latest or a specific incident\n"
                        "• [bold cyan]/report [run_id][/bold cyan] — Open or generate a run's HTML report\n"
                        "  [dim]/run, /fix and /report take the same flags as the CLI — add --help to list them[/dim]\n"
                        "• [bold cyan]/model[/bold cyan] — View, switch, or add AI models (e.g. /model add local ollama/llama3)\n"
                        "• [bold cyan]/runs[/bold cyan] — List recent test runs, crashes, and report files\n"
                        "• [bold cyan]/rules[/bold cyan] — Display active business rules from .loki/rules.md\n"
                        "• [bold cyan]/update[/bold cyan] — Check for updates and upgrade LOKI via uv\n"
                        "• [bold cyan]/clear[/bold cyan] — Clear terminal screen\n"
                        "• [bold cyan]exit[/bold cyan] / [bold cyan]quit[/bold cyan] — Exit interactive chat session\n"
                        "• Ask any question about QA testing, code bugs, race conditions, or past runs!",
                        title="💡 LOKI Chat Help",
                        border_style="cyan",
                    )
                )
                continue

            if cmd_lower in ["/update", "/upgrade"]:
                self._handle_update_command()
                continue

            if cmd_lower == "/model" or cmd_lower.startswith("/model "):
                self._handle_model_command(user_input[len("/model"):].strip())
                continue

            cli_command = next(
                (c for c in CLI_SLASH_COMMANDS if cmd_lower == f"/{c}" or cmd_lower.startswith(f"/{c} ")),
                None,
            )
            if cli_command:
                self._invoke_cli_command(cli_command, user_input[len(cli_command) + 1:].strip())
                continue

            if cmd_lower == "/runs":
                self._display_runs_table()
                continue

            if cmd_lower == "/rules":
                self._display_rules()
                continue

            if cmd_lower == "/clear":
                os.system("cls" if os.name == "nt" else "clear")
                continue

            # Multi-turn conversational query
            self.history.append({"role": "user", "content": user_input})

            # Prevent context bloat by retaining only recent conversation turns
            active_messages = self.history
            if len(self.history) > 12:
                # Keep system prompt at index 0, take last 10 messages
                active_messages = [self.history[0]] + self.history[-10:]

            success = False
            try:
                # Wait for the first token inside the spinner, then stream text cleanly to stdout
                with Status("[bold yellow]LOKI is thinking...[/bold yellow]", console=self.console):
                    chunks = llm.open_stream(active_messages, model=self._explicit_model)
                self.console.print()
                collected = []
                for text in chunks:
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    collected.append(text)
                sys.stdout.write("\n")
                sys.stdout.flush()

                self.history.append({"role": "assistant", "content": "".join(collected)})
                success = True
            except llm.LLMUnavailableError as e:
                self.console.print(f"\n[bold red]AI Error:[/bold red] {escape(e.summary)}. Please try again in a moment.")
            except Exception as e:
                error = e if isinstance(e, llm.LLMError) else llm.LLMError(str(e), e)
                self.console.print(f"\n[bold red]AI Error:[/bold red] {escape(error.summary)}")
                if error.is_rate_limited:
                    self.console.print("[dim]Wait for the quota/rate limit to reset, or select an available model with /model.[/dim]")

            if not success and self.history and self.history[-1].get("role") == "user":
                self.history.pop()
