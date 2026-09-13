#!/usr/bin/env python3
"""Datasets Factory — pipeline complet de génération, QA, packaging.

Usage:
    python3 factory.py generate <topic.json>   # génère le dataset depuis une spec
    python3 factory.py qa <dataset.jsonl>      # contrôle qualité
    python3 factory.py package <name>          # package JSONL + README
"""
import json
import os
import re
import sys
import time
import hashlib
import urllib.request
from pathlib import Path

FACTORY = Path.home() / "Projets" / "datasets-factory"
DATASETS_DIR = FACTORY / "datasets"
OUTPUT_DIR = FACTORY / "output"
API_URL = "http://localhost:20128/v1/chat/completions"
MODEL = "cl/cline-free/solar-pro4"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def call_llm(messages: list, max_tokens: int = 1500, temperature: float = 0.8,
             retries: int = 3) -> str | None:
    """Appelle Solar Pro 4 via 9router avec retry."""
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                API_URL,
                headers={"Content-Type": "application/json"},
                data=payload,
            )
            resp = json.load(urllib.request.urlopen(req, timeout=120))
            content = resp["choices"][0]["message"]["content"]
            if content and content.strip():
                return content.strip()
        except Exception as e:
            wait = 10 * (attempt + 1)
            print(f"  [retry {attempt+1}/{retries}] {e} — attente {wait}s")
            time.sleep(wait)
    return None


def parse_json_block(text: str):
    """Parse le JSON d'une réponse LLM, tolérant aux fences markdown."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:]
            if p.strip().startswith("[") or p.strip().startswith("{"):
                text = p.strip()
                break
    # Première tentative : tel quel
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Deuxième : extraire le tableau le plus externe
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def validate_example(ex) -> bool:
    """Validation structurelle stricte d'un exemple."""
    if not isinstance(ex, dict):
        return False
    q, a = ex.get("q"), ex.get("a")
    if not q or not a:
        return False
    if not isinstance(q, str) or not isinstance(a, str):
        return False
    # Question : 15-300 chars, doit finir par ? (Q&A format)
    if not (15 <= len(q) <= 300):
        return False
    # Réponse : 30-1500 chars, phrases complètes
    if not (30 <= len(a) <= 1500):
        return False
    if not q.strip().endswith("?"):
        return False
    # Pas de placeholder / garbage
    garbage = ["lorem", "xxx", "TODO", "exemple de", "[insérer", "etc..."]
    low = (q + a).lower()
    if any(g in low for g in garbage):
        return False
    return True


def gen_batch(spec: dict, batch_size: int = 10) -> list:
    """Génère un lot d'exemples selon la spec."""
    prompt = f"""Tu es un expert en création de données d'entraînement pour LLM.
Génère {batch_size} exemples de paires question/réponse en {spec.get('lang', 'français')}.

DOMAINE : {spec['domain']}
NIVEAU : {spec.get('level', 'intermédiaire')}
STYLE : {spec.get('style', 'questions techniques précises, réponses factuelles complètes de 2-4 phrases')}

Exigences qualité :
- Questions variées (definition, procédure, sécurité, diagnostic, normes...)
- Réponses SANS hésitation, factuelles, précises, autonomes (compréhensibles sans la question)
- Pas de "ça dépend", pas de références à d'autres questions
- Terminologie professionnelle correcte du domaine

Format : réponds UNIQUEMENT avec le tableau JSON strict :
[{{"q": "...", "a": "..."}}, ...]"""
    content = call_llm([{"role": "user", "content": prompt}])
    if not content:
        return []
    data = parse_json_block(content)
    if not isinstance(data, list):
        return []
    return [ex for ex in data if validate_example(ex)]


def qa_check(dataset: list) -> tuple[dict, list]:
    """Contrôle qualité réel : dédup, diversité, stats."""
    seen_q, seen_h, unique = set(), set(), []
    for ex in dataset:
        q_norm = re.sub(r"\s+", " ", ex["q"].strip().lower())
        q_hash = hashlib.md5(q_norm.encode()).hexdigest()
        if q_hash in seen_q:
            continue
        # near-dup : 8 premiers mots
        sig = " ".join(q_norm.split()[:8])
        if sig in seen_h:
            continue
        seen_q.add(q_hash)
        seen_h.add(sig)
        unique.append(ex)

    qlens = [len(e["q"]) for e in unique]
    alens = [len(e["a"]) for e in unique]
    vocab = len(set(" ".join(e["a"] for e in unique).lower().split()))
    report = {
        "total_raw": len(dataset),
        "unique": len(unique),
        "dup_removed": len(dataset) - len(unique),
        "q_len_avg": round(sum(qlens) / max(len(qlens), 1), 1),
        "a_len_avg": round(sum(alens) / max(len(alens), 1), 1),
        "answer_vocab": vocab,
        "diversity_score": round(vocab / max(sum(alens), 1) * 1000, 2),
    }
    return report, unique


def cmd_generate(spec_path: str):
    spec = json.loads(Path(spec_path).read_text())
    name = spec["name"]
    target = spec.get("target", 250)
    batch_size = spec.get("batch_size", 10)
    print(f"=== GÉNÉRATION : {name} — cible {target} exemples ===")

    raw_path = DATASETS_DIR / f"{name}_raw.jsonl"
    all_examples = []
    if raw_path.exists():
        all_examples = [json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()]
        print(f"  reprise : {len(all_examples)} exemples déjà générés")

    batch_n = 0
    while len(all_examples) < target:
        batch_n += 1
        print(f"  lot {batch_n} ({len(all_examples)}/{target})...", end=" ", flush=True)
        t0 = time.time()
        examples = gen_batch(spec, batch_size)
        dt = time.time() - t0
        print(f"{len(examples)} valides en {dt:.0f}s")
        all_examples.extend(examples)
        # checkpoint
        with raw_path.open("w") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        time.sleep(2)  # anti rate-limit

    # QA finale
    report, unique = qa_check(all_examples)
    final_path = DATASETS_DIR / f"{name}.jsonl"
    with final_path.open("w") as f:
        for ex in unique:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    report["final_file"] = str(final_path)
    (DATASETS_DIR / f"{name}_qa_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_qa(dataset_path: str):
    data = [json.loads(l) for l in Path(dataset_path).read_text().splitlines() if l.strip()]
    report, unique = qa_check(data)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_package(name: str):
    final = DATASETS_DIR / f"{name}.jsonl"
    spec_candidates = list(DATASETS_DIR.glob(f"{name}_spec.json")) + \
                      list(FACTORY.glob(f"{name}_spec.json"))
    spec = {}
    if spec_candidates:
        spec = json.loads(spec_candidates[0].read_text())
    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = final.read_text().splitlines()
    n = len(lines)
    # train split 90% / sample 10%
    split = int(n * 0.9)
    (out_dir / "train.jsonl").write_text("\n".join(lines[:split]) + "\n")
    (out_dir / "sample.jsonl").write_text("\n".join(lines[split:]) + "\n")

    readme = f"""# {name}

Dataset SFT (question/réponse) généré et validé automatiquement.

- **Domaine** : {spec.get('domain', '—')}
- **Langue** : {spec.get('lang', 'français')}
- **Niveau** : {spec.get('level', 'intermédiaire')}
- **Exemples** : {n} (train {split} + sample {n - split})
- **Format** : JSONL `{{"q": "...", "a": "..."}}`
- **Licence** : CC-BY-4.0

## Contrôle qualité
- Déduplication stricte (hash question + signature 8 mots)
- Validation structurelle (longueurs, format, absence de placeholders)
- Aucune donnée personnelle

## Achat / Support
Paiement : https://payrequest.me/lvs0
Contact : relay-lvs0@protonmail.com
"""
    (out_dir / "README.md").write_text(readme)
    print(f"✅ Packagé dans {out_dir} — {n} exemples")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "generate":
        cmd_generate(sys.argv[2])
    elif cmd == "qa":
        cmd_qa(sys.argv[2])
    elif cmd == "package":
        cmd_package(sys.argv[2])
    else:
        print(__doc__)
