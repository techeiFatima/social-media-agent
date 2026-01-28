"""
Helper: discover the correct TELEGRAM_CHAT_ID for your bot.

Steps:
1) Open Telegram and message your bot (press "Start", send "hi")
2) Run:
   python3 "workshop 2.2/telegram_get_chat_id.py"
3) Copy the printed chat_id into your root `.env` as TELEGRAM_CHAT_ID
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from telegram import Bot


load_dotenv(Path(__file__).parent.parent / ".env")


async def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN in .env")

    bot = Bot(token=token)
    updates = await bot.get_updates(limit=25)

    print(f"updates={len(updates)}")
    if not updates:
        print("No updates yet. Open Telegram, press 'Start' on your bot, and send any message, then run again.")
        return

    print("\nFound chats (copy the chat_id you want):")
    seen = set()
    for u in updates:
        m = u.effective_message
        if not m:
            continue
        chat = m.chat
        if chat.id in seen:
            continue
        seen.add(chat.id)

        user = m.from_user
        username = getattr(user, "username", None)
        first_name = getattr(user, "first_name", None)
        chat_type = getattr(chat, "type", None)
        print(f"- chat_id={chat.id} chat_type={chat_type} from={username or first_name}")


if __name__ == "__main__":
    asyncio.run(main())

