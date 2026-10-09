"""Cloze (likelihood) matched-guise probe for open-weights causal LMs.
 
For each stimulus text and trait, scores log P(trait | prompt with text),
then subtracts the log P(trait | prompt with a content-free baseline text),
so results show how much the dialect *shifts* each trait, not the trait's prior.
 
Usage:
    python cloze_probe.py --model meta-llama/Llama-3.1-8B --out cloze_llama.csv
"""
import argparse
import csv

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

@torch.no_grad()
def trait_logprob(model, tok, prefix:str, trait:str) -> tuple[float, int]:
   """Sum log-prob of the trait tokens given the prefix. Returns (log_prob, num_tokens). """ 
   prefix_ids = tok(prefix, return_tensors="pt").input_ids
   trait_ids = tok(trait, return_tensors="pt").input_ids
   ids = torch.cat([prefix_ids, trait_ids], dim=1).to(model.device)
   logits = model(ids).logits[0, :-1]
   logprobs = torch.log_softmax(logits, dim=-1)
   n_prefix = prefix_ids.shape[1]
   targets = ids[0, 1:]
   trait_positions = range(n_prefix - 1, ids.shape[1] - 1)
   lp = sum(logprobs[p, targets[p]].item() for p in trait_positions)
   return lp, trait_ids.shape[1]

def split_template(template: str, text: str) -> str:
    """Return the template up to the {trait} slot, with text filled in."""
    head = template.split("{trait}")[0]
    return head.format(text=text)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stimuli", default=config.STIMULI_CSV)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
 
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
 
    df = pd.read_csv(args.stimuli)
    rows = []
 
    # Baselines: prior log-prob per (template, trait), averaged over baseline texts.
    baseline = {}
    for ti, tmpl in enumerate(config.CLOZE_TEMPLATES):
        for trait in config.TRAITS:
            vals = [
                trait_logprob(model, tok, split_template(tmpl, bt), trait)[0]
                for bt in config.BASELINE_TEXTS
            ]
            baseline[(ti, trait)] = sum(vals) / len(vals)
 
    for _, r in df.iterrows():
        for variety, col in (("pidgin", "pidgin_text"), ("british", "british_text")):
            text = r[col]
            n_tok = len(tok(text, add_special_tokens=False).input_ids)
            for ti, tmpl in enumerate(config.CLOZE_TEMPLATES):
                prefix = split_template(tmpl, text)
                for trait in config.TRAITS:
                    lp, n_trait_tok = trait_logprob(model, tok, prefix, trait)
                    rows.append({
                        "model": args.model, "item_id": r["item_id"],
                        "variety": variety, "template": ti, "trait": trait,
                        "logprob": lp, "baseline_logprob": baseline[(ti, trait)],
                        "shift": lp - baseline[(ti, trait)],
                        "n_text_tokens": n_tok, "n_trait_tokens": n_trait_tok,
                    })
 
    pd.DataFrame(rows).to_csv(args.out, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"wrote {len(rows)} rows to {args.out}")
 
 
if __name__ == "__main__":
    main()
 