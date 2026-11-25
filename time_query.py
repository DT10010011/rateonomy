"""Ask Gemini for the current local time."""

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai


load_dotenv(Path(__file__).with_name(".env"))


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "NONE":
        raise RuntimeError("Set GEMINI_API_KEY in the .env file before running this script.")

    current_time = datetime.now().astimezone().isoformat(timespec="seconds")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=(
            "What time is it right now? Use this current local time from my computer "
            f"as the source of truth: {current_time}"
        ),
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    print(response.text)


if __name__ == "__main__":
    main()
