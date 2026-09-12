"""
Fetch real citations for the biomechanics grounding via the Grok API.

Requires: pip install requests
Requires: an XAI_API_KEY env var (from the hackathon's participant credits)

Note: verify the model name against xAI's current docs before running --
API model strings change over time and I can't check this from here
(api.x.ai isn't reachable from my sandbox to test this directly).
"""

import os

import requests

XAI_API_KEY = os.environ.get("XAI_API_KEY")
if not XAI_API_KEY:
    raise SystemExit("Set the XAI_API_KEY environment variable first.")

MODEL = "grok-4"  # double check this against xAI's current docs

PROMPT = """
Find real, verifiable peer-reviewed papers on these topics for a rock
climbing biomechanics project:

1. Ape index (arm span to height ratio) and rock climbing performance
2. Kinanthropometric profiling of climbers (anthropometrics vs. climbing grade)
3. Center-of-mass kinematics / movement efficiency in climbing

For each paper, give: exact title, authors, year, journal, and DOI if you
have one. Only include papers you are confident actually exist -- if
you're not sure a specific paper is real, say so explicitly rather than
guessing at plausible-sounding details.
"""


def ask_grok(prompt, model=MODEL):
    resp = requests.post(
        "https://api.x.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {XAI_API_KEY}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print(ask_grok(PROMPT))
