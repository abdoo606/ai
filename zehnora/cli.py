"""Desktop terminal mode: talk to Zehnora in the console.

    python -m zehnora            # interactive
    python -m zehnora "what time is it"   # one question
"""

from __future__ import annotations

import os
import sys

from .agent import Assistant


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    bot = Assistant.from_settings()
    if argv:
        print(bot.ask(" ".join(argv)))
        return 0
    print("Zehnora is online. Type 'exit' to quit, 'reset' to clear the conversation.")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if text.lower() in ("exit", "quit", "خروج"):
            return 0
        if text.lower() == "reset":
            bot.reset()
            print("conversation cleared")
            continue
        if text:
            print(f"zehnora> {bot.ask(text)}")


def serve() -> None:
    """Run the packaged web server, loopback-only unless explicitly configured."""
    import uvicorn

    host = os.environ.get("ZEHNORA_HOST", "127.0.0.1")
    port = int(os.environ.get("ZEHNORA_PORT", "8000"))
    uvicorn.run("zehnora.server:get_app", factory=True, host=host, port=port)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
