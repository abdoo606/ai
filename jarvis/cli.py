"""Desktop terminal mode: talk to Jarvis in the console.

    python -m jarvis            # interactive
    python -m jarvis "what time is it"   # one question
"""

from __future__ import annotations

import sys

from .agent import Assistant


def main(argv: list[str]) -> int:
    bot = Assistant.from_settings()
    if argv:
        print(bot.ask(" ".join(argv)))
        return 0
    print("Jarvis is online. Type 'exit' to quit, 'reset' to clear the conversation.")
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
            print(f"jarvis> {bot.ask(text)}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
