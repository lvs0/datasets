#!/usr/bin/env python3
"""Datasets Factory v2 — format enrichi: q/context/a/difficulty/tags.

Format cible (style linux-fr-support-synthetic-v1):
{"q": "...", "context": "...", "a": "...", "difficulty": "débutant|intermédiaire|avancé",
 "tags": ["réseau", "systemd", ...]}

Usage:
    python3 factory2.py generate <spec.json>
    python3 factory2.py qa <dataset.jsonl>
    python3 factory2.py package <name> [--sample N]
"""
import json
import os
import re
import sys
import time
import hashlib
import random
import urllib.request
from pathlib import Path

FACTORY = Path.home() / "Projets" / "datasets-factory"
DATASETS_DIR = FACTORY / "datasets"
OUTPUT_DIR = FACTORY / "output"
API_URL = "http://localhost:20128/v1/chat/completions"
MODEL = "cl/cline-free/solar-pro4"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_DIFFICULTY = {"débutant", "intermédiaire", "avancé"}


def call_llm(messages: list, max_tokens: int = 2000, temperature: float = 0.85,
             retries: int = 3) -> str | None:
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
            resp = json.load(urllib.request.urlopen(req, timeout=150))
            content = resp["choices"][0]["message"]["content"]
            if content and content.strip():
                return content.strip()
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f"    [retry {attempt+1}/{retries}] {type(e).__name__} — attente {wait}s", flush=True)
            time.sleep(wait)
    return None


def parse_json_block(text: str):
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
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def normalize_example(ex: dict) -> dict | None:
    """Normalise et valide un exemple au format enrichi."""
    if not isinstance(ex, dict):
        return None
    # Tolérer les deux formats (ancien q/a et nouveau)
    q = ex.get("q") or ex.get("question")
    a = ex.get("a") or ex.get("answer") or ex.get("réponse")
    ctx = ex.get("context") or ex.get("contexte") or ""
    diff = ex.get("difficulty") or ex.get("difficulté") or "intermédiaire"
    tags = ex.get("tags") or []

    if not q or not a:
        return None
    q, a, ctx = str(q).strip(), str(a).strip(), str(ctx).strip()
    if not (15 <= len(q) <= 400):
        return None
    if not (30 <= len(a) <= 2000):
        return None

    diff = diff.strip().lower()
    if diff not in VALID_DIFFICULTY:
        diff = "intermédiaire"

    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip().lower() for t in tags if str(t).strip()][:6]

    return {"q": q, "context": ctx, "a": a, "difficulty": diff, "tags": tags}


def validate_example(ex: dict) -> bool:
    if not ex:
        return False
    q, a, ctx = ex["q"], ex["a"], ex["context"]
    if not q.strip().endswith("?"):
        return False
    garbage = ["lorem", "xxx", "todo", "[insérer", "exemple de ", "etc...", "votre texte"]
    low = (q + " " + a + " " + ctx).lower()
    if any(g in low for g in garbage):
        return False
    # Réponse autonome: commence par une majuscule
    if a and a[0].islower():
        return False
    return True


def gen_batch(spec: dict, category: str, focus: str, batch_size: int = 8) -> list:
    categories = ", ".join(spec["categories"])
    prompt = f"""Tu es un expert {spec['domain_expert']} qui crée des données d'entraînement pour LLM.
Génère {batch_size} cas en {spec.get('lang', 'français')} sur la catégorie « {category} ».

CONSIGNE CATÉGORIE : {focus}

Cadre général du dataset : {spec['domain']}

Exigences :
- Chaque cas = un problème/réel ou une question technique avec du CONTEXTE (situation, message d'erreur, config)
- Question précise et autonome
- Contexte : 1-2 phrases décrivant la situation (peut être vide si question générale)
- Réponse : complète, actionnable, factuelle, 2-5 phrases, commence par une majuscule
- difficulty : "débutant", "intermédiaire" ou "avancé" selon la complexité réelle
- tags : 2-4 mots-clés en minuscule

Réponds UNIQUEMENT avec le tableau JSON strict :
[{{"q": "...", "context": "...", "a": "...", "difficulty": "...", "tags": ["..."]}}, ...]"""
    content = call_llm([{"role": "user", "content": prompt}])
    if not content:
        return []
    data = parse_json_block(content)
    if not isinstance(data, list):
        return []
    out = []
    for ex in data:
        norm = normalize_example(ex)
        if norm and validate_example(norm):
            norm["category"] = category
            out.append(norm)
    return out


def qa_check(dataset: list) -> tuple[dict, list]:
    seen_q, seen_h, unique = set(), set(), []
    for ex in dataset:
        q_norm = re.sub(r"\s+", " ", ex["q"].strip().lower())
        q_hash = hashlib.md5(q_norm.encode()).hexdigest()
        if q_hash in seen_q:
            continue
        sig = " ".join(q_norm.split()[:8])
        if sig in seen_h:
            continue
        seen_q.add(q_hash)
        seen_h.add(sig)
        unique.append(ex)

    # Équilibre des catégories
    cat_counts = {}
    diff_counts = {}
    for e in unique:
        cat_counts[e["category"]] = cat_counts.get(e["category"], 0) + 1
        diff_counts[e["difficulty"]] = diff_counts.get(e["difficulty"], 0) + 1

    alens = [len(e["a"]) for e in unique]
    vocab = len(set(" ".join(e["a"] for e in unique).lower().split()))
    report = {
        "total_raw": len(dataset),
        "unique": len(unique),
        "dup_removed": len(dataset) - len(unique),
        "a_len_avg": round(sum(alens) / max(len(alens), 1), 1),
        "answer_vocab": vocab,
        "diversity_score": round(vocab / max(sum(alens), 1) * 1000, 2),
        "categories": cat_counts,
        "difficulties": diff_counts,
    }
    return report, unique


def cmd_generate(spec_path: str):
    spec = json.loads(Path(spec_path).read_text())
    name = spec["name"]
    target = spec.get("target", 5000)
    batch_size = spec.get("batch_size", 8)
    print(f"=== GÉNÉRATION v2 : {name} — cible {target} ===", flush=True)

    raw_path = DATASETS_DIR / f"{name}_raw.jsonl"
    all_examples = []
    if raw_path.exists():
        all_examples = [json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()]
        print(f"  reprise : {len(all_examples)} exemples existants", flush=True)

    cat_counts = {}
    for e in all_examples:
        cat_counts[e["category"]] = cat_counts.get(e["category"], 0) + 1

    # Rotation des catégories : celle qui a le moins d'exemples d'abord
    batch_n = 0
    empty_streak = 0
    while len(all_examples) < target and empty_streak < 6:
        batch_n += 1
        # Catégorie sous-représentée
        remaining_cats = [c for c in spec["categories"]
                          if cat_counts.get(c, 0) < target / len(spec["categories"]) * 1.2]
        cat = random.choice(remaining_cats) if remaining_cats else random.choice(spec["categories"])
        focus = spec.get("category_focus", {}).get(cat, "")
        print(f"  lot {batch_n} [{cat}] ({len(all_examples)}/{target})...", end=" ", flush=True)
        t0 = time.time()
        examples = gen_batch(spec, cat, focus, batch_size)
        dt = time.time() - t0
        print(f"{len(examples)} valides en {dt:.0f}s", flush=True)
        if examples:
            all_examples.extend(examples)
            for e in examples:
                cat_counts[e["category"]] = cat_counts.get(e["category"], 0) + 1
            empty_streak = 0
        else:
            empty_streak += 1
        # Checkpoint
        with raw_path.open("w") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        time.sleep(2)

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


def cmd_package(name: str, sample_n: int = 250):
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
    sample_n = min(sample_n, n)
    # Échantillon stratifié : proportionnel par catégorie
    by_cat = {}
    for i, line in enumerate(lines):
        ex = json.loads(line)
        by_cat.setdefault(ex["category"], []).append(line)
    sample_lines = []
    for cat, cat_lines in by_cat.items():
        k = max(1, round(sample_n * len(cat_lines) / n))
        sample_lines.extend(random.sample(cat_lines, min(k, len(cat_lines))))
    random.shuffle(sample_lines)
    sample_lines = sample_lines[:sample_n]
    sampled_idx = {id(l) for l in sample_lines}
    train_lines = [l for l in lines if l not in set(sample_lines)]

    (out_dir / "train.jsonl").write_text("\n".join(train_lines) + "\n")
    (out_dir / "sample.jsonl").write_text("\n".join(sample_lines) + "\n")

    report_path = DATASETS_DIR / f"{name}_qa_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    readme = f"""# {name}

Dataset SFT {spec.get('lang', 'français')} — cas enrichis (question, contexte, réponse, difficulté, tags).

- **Domaine** : {spec.get('domain', '—')}
- **Catégories** : {', '.join(spec.get('categories', []))}
- **Exemples** : {n} ({len(train_lines)} train + {len(sample_lines)} échantillon gratuit)
- **Format** : JSONL `{{"q", "context", "a", "difficulty", "tags", "category"}}`
- **Licence** : commerciale limitée à l'entraînement et aux tests
- **Données** : 100% synthétiques, aucune donnée personnelle

## Qualité mesurée
{json.dumps({k: v for k, v in report.items() if k != 'final_file'}, indent=2, ensure_ascii=False)}

## Méthode de génération
Génération par un LLM (Solar Pro 4), par lots avec rotation de catégories, puis
validation automatisée : déduplication stricte, contrôle de structure, détection
de placeholders, vérification de l'autonomie des réponses.

## Limites
- Données générées par IA : vérification humaine recommandée avant usage critique
- Réponses non garanties à jour des dernières versions du domaine

## Achat
Paiement : https://payrequest.me/lvs0
Contact : relay-lvs0@protonmail.com
Livraison du fichier complet sous 24 h après paiement.
"""
    (out_dir / "README.md").write_text(readme)
    print(f"✅ Packagé : {out_dir} — train {len(train_lines)}, sample {len(sample_lines)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "generate":
        cmd_generate(sys.argv[2])
    elif cmd == "qa":
        cmd_qa(sys.argv[2])
    elif cmd == "package":
        name = sys.argv[2]
        sample_n = 250
        if "--sample" in sys.argv:
            sample_n = int(sys.argv[sys.argv.index("--sample") + 1])
        cmd_package(name, sample_n)
    else:
        print(__doc__)
