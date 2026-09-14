#!/bin/bash
# Autopilot datasets-factory : surveille la génération, package + publie quand fini
cd /home/l-vs/Projets/datasets-factory

NAME=linux-fr-support-synthetic-v1
TARGET=5000
RAW=datasets/${NAME}_raw.jsonl
LOG=logs/gen-linux.log

while true; do
  # La génération a-t-elle terminé ?
  if grep -q "GEN_DONE" "$LOG" 2>/dev/null; then
    echo "[autopilot] génération terminée détectée"
    break
  fi
  # Le process de génération est-il vivant ?
  if ! pgrep -f "factory2.py generate" > /dev/null; then
    # Peut-être tombé — relancer si sous la cible
    N=$(wc -l < "$RAW" 2>/dev/null || echo 0)
    if [ "$N" -lt "$TARGET" ]; then
      echo "[autopilot] process mort à $N/$TARGET — relance" 
      nohup python3 factory2.py generate ${NAME}_spec.json >> "$LOG" 2>&1 &
    else
      echo "[autopilot] cible atteinte ($N)"
      echo "GEN_DONE exit=0" >> "$LOG"
      break
    fi
  fi
  sleep 300
done

# QA + package + publication
N=$(wc -l < "$RAW" 2>/dev/null || echo 0)
if [ "$N" -ge 3000 ]; then
  echo "[autopilot] QA + packaging de $N exemples"
  python3 factory2.py package $NAME --sample 250
  # Publication GitHub (train + sample + QA report + README)
  git add -A
  git commit -m "Dataset ${NAME}: ${N} exemples validés + échantillon public 250" -q
  git push -q origin main 2>&1 | grep -v Everything || true
  echo "[autopilot] publié sur github.com/lvs0/datasets"
else
  echo "[autopilot] trop peu d'exemples ($N) — pas de publication"
fi
echo "[autopilot] FINI"
