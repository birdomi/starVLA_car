#!/usr/bin/env bash
# Convert every RoboTwin 2.0 aloha-agilex task (clean_50 + randomized_500) into LeRobot v2.0 datasets.
# Waits for each zip to appear (so it can run concurrently with the HF snapshot download).
# Usage: bash convert_all_robotwin.sh [raw_dataset_dir] [out_root] [workers]
set -uo pipefail
RAW="${1:-/mnt/DATA2/yyg/RoboTwin2.0/dataset}"
OUT="${2:-/mnt/DATA2/yyg/RoboTwin2.0/lerobot}"
WORKERS="${3:-16}"
PY="${STARVLA_PYTHON:-$HOME/miniconda3/envs/starVLA/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROBOTWIN_EXTRACT_TMP="${ROBOTWIN_EXTRACT_TMP:-$(dirname "$OUT")/_extract}"
mkdir -p "$ROBOTWIN_EXTRACT_TMP" "$OUT/Clean" "$OUT/Randomized"

TASKS=(adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block handover_mic hanging_mug lift_pot move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad open_laptop open_microwave pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right place_bread_basket place_bread_skillet place_burger_fries place_can_basket place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup place_fan place_mouse_pad place_object_basket place_object_scale place_object_stand place_phone_stand place_shoe press_stapler put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object shake_bottle shake_bottle_horizontally stack_blocks_three stack_blocks_two stack_bowls_three stack_bowls_two stamp_seal turn_switch)

for split in "Clean:aloha-agilex_clean_50" "Randomized:aloha-agilex_randomized_500"; do
  name="${split%%:*}"; zipbase="${split##*:}"
  for t in "${TASKS[@]}"; do
    zip="$RAW/$t/$zipbase.zip"; out="$OUT/$name/$t"
    if [[ -f "$out/meta/info.json" ]]; then echo "[skip] $name/$t"; continue; fi
    while [[ ! -f "$zip" ]]; do echo "[wait] $zip"; sleep 120; done
    echo "[convert] $name/$t  $(date)"
    "$PY" "$SCRIPT_DIR/convert_robotwin_to_lerobot.py" --zip "$zip" --out "$out" --workers "$WORKERS" \
      || { echo "[FAIL] $name/$t"; rm -rf "$out"; }
  done
done
echo "ALL_CONVERT_DONE $(date)"
