# 🗂️ Datasets Factory

Datasets SFT **synthétiques, enrichis et validés automatiquement** pour fine-tuner vos LLM — en français.

## 📦 Produit phare : `linux-fr-support-synthetic-v1`

**5000 cas de support Linux en français** pour entraîner des assistants de dépannage.

| Attribut | Valeur |
|---|---|
| Format | JSONL `{"q", "context", "a", "difficulty", "tags", "category"}` |
| Catégories | réseau, systemd, paquets, permissions, stockage |
| Difficultés | débutant / intermédiaire / avancé |
| Licence | commerciale limitée à l'entraînement et aux tests |
| Données | 100% synthétiques, zéro donnée personnelle |
| Échantillon gratuit | 250 lignes (`sample.jsonl`) |

### 💰 Acheter — 49 € (paiement unique, crypto ou carte)

**[payrequest.me/lvs0](https://payrequest.me/lvs0)** — précisez `linux-fr-support-synthetic-v1` dans le message.
Livraison du `train.jsonl` complet sous 24 h (email ou lien privé).

### Licence acheteur
- ✅ Entraînement de vos modèles (interne ou commercial)
- ✅ Tests et évaluations
- ❌ Revente/redistribution du dataset en l'état
- ❌ Inclusion dans un dataset public

## 🔍 Qualité mesurée (pas du marketing)

Pipeline automatisé dans ce repo (`factory2.py`) :

1. **Validation structurelle** — JSON strict, longueurs bornées, questions complètes, réponses autonomes
2. **Détection de placeholders** — aucun « lorem », « TODO », etc.
3. **Déduplication stricte** — hash question + signature near-dup (8 premiers mots)
4. **Statistiques publiées** — équilibre des catégories, distribution des difficultés, vocabulaire des réponses

Le rapport QA complet est publié : `datasets/linux-fr-support-synthetic-v1_qa_report.json`.

### Limites (honnêteté)
- Données générées par IA (Solar Pro 4) : une vérification humaine par échantillonnage est recommandée avant usage critique
- Les réponses ne sont pas garanties à jour des toutes dernières versions des distributions

## 🛠️ Reproduire (open source, MIT)

```bash
python3 factory2.py generate <spec.json>   # génération par lots, rotation catégories, checkpoint
python3 factory2.py qa <file.jsonl>        # contrôle qualité + stats
python3 factory2.py package <name>         # packaging train/sample/README (échantillon stratifié)
```

Coût marginal de génération ≈ 0 (modèle gratuit via passerelle locale).

## 📄 Licences

- Code de la factory : MIT
- Datasets : licence commerciale limitée (voir README de chaque dataset)

---
Contact : relay-lvs0@protonmail.com · [payrequest.me/lvs0](https://payrequest.me/lvs0)
