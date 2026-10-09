"""Likert + forced-choice matched-guise probe for chat/API models.

Uses the Anthropic SDK by default (set ANTHROPIC_API_KEY); swap `ask()` for
another provider if you want to compare models. Temperature 0, several repeats.

Usage:
    python likert_probe.py --model <model-id> --out likert_<name>.csv --repeats 3
"""
import argparse
import random
import re

import anthropic
import pandas as pd

import config

client = anthropic.Anthropic()


def ask(model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 8) -> str:
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def parse_int(s: str):
    m = re.search(r"[1-7]", s)
    return int(m.group()) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stimuli", default=config.STIMULI_CSV)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    df = pd.read_csv(args.stimuli)
    rows = []

    for _, r in df.iterrows():
        for variety, col in (("pidgin", "pidgin_text"), ("british", "british_text")):
            for dim in config.LIKERT_DIMENSIONS:
                for cond, tmpl in (("plain", config.LIKERT_PROMPT),
                                   ("control", config.LIKERT_PROMPT_CONTROL)):
                    prompt = tmpl.format(text=r[col], dimension=dim)
                    for rep in range(args.repeats):
                        raw = ask(args.model, prompt)
                        rows.append({
                            "model": args.model, "task": "likert", "item_id": r["item_id"],
                            "variety": variety, "trait": dim, "condition": cond,
                            "repeat": rep, "raw": raw, "value": parse_int(raw),
                        })

        # Forced choice: randomize which variety is shown as Person A.
        for trait in config.LIKERT_DIMENSIONS:
            pidgin_first = random.random() < 0.5
            a, b = (r["pidgin_text"], r["british_text"]) if pidgin_first else (r["british_text"], r["pidgin_text"])
            prompt = config.FORCED_CHOICE_PROMPT.format(a=a, b=b, trait=trait)
            for rep in range(args.repeats):
                raw = ask(args.model, prompt)
                letter = re.search(r"\b([AB])\b", raw)
                letter = letter.group(1) if letter else None
                chose_pidgin = None if letter is None else ((letter == "A") == pidgin_first)
                rows.append({
                    "model": args.model, "task": "forced_choice", "item_id": r["item_id"],
                    "variety": "pair", "trait": trait, "condition": "plain",
                    "repeat": rep, "raw": raw,
                    "value": None if chose_pidgin is None else int(chose_pidgin),
                    "pidgin_first": int(pidgin_first),
                })

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()