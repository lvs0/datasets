# linux-fr-support-synthetic-v1

Dataset SFT français — cas enrichis (question, contexte, réponse, difficulté, tags).

- **Domaine** : administration et dépannage Linux (distributions Debian/Ubuntu/Fedora/Arch)
- **Catégories** : réseau, systemd, paquets, permissions, stockage
- **Exemples** : 3759 (3509 train + 250 échantillon gratuit)
- **Format** : JSONL `{"q", "context", "a", "difficulty", "tags", "category"}`
- **Licence** : commerciale limitée à l'entraînement et aux tests
- **Données** : 100% synthétiques, aucune donnée personnelle

## Qualité mesurée
{
  "total_raw": 5007,
  "unique": 3759,
  "dup_removed": 1248,
  "a_len_avg": 318.9,
  "answer_vocab": 12347,
  "diversity_score": 10.3,
  "categories": {
    "systemd": 857,
    "réseau": 765,
    "paquets": 820,
    "permissions": 706,
    "stockage": 611
  },
  "difficulties": {
    "débutant": 1025,
    "intermédiaire": 1646,
    "avancé": 1088
  }
}

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
