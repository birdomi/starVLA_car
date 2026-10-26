#!/bin/bash
# Wait for the grouprope run (GPUs 0-3) to save steps_100000 and exit, then launch the progress-shift run on 0-3.
cd /mnt/DATA2/yyg/starVLA
CK=playground/Checkpoints/starvla_qwengr00t_actionmap_grouprope_carturn_0p8b_3cam_abs/checkpoints/steps_100000_pytorch_model.pt
YAML=examples/realRobots/H1_2_car_real/train_files/starvla_qwengr00t_actionmap_grouprope_progshift_carturn_0p8b_3cam_abs.yaml
echo "[$(date +%T)] waiting for $CK and grouprope exit"
until [ -f "$CK" ] && ! pgrep -f "config_yaml .*starvla_qwengr00t_actionmap_grouprope_carturn_0p8b_3cam_abs.yaml" >/dev/null; do sleep 30; done
echo "[$(date +%T)] grouprope done; waiting for GPUs 0-3 to free"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3 | awk '$1>4000' | wc -l)" = 0 ]; do sleep 20; done
python3 -c "import torch,sys; torch.load(sys.argv[1], map_location='cpu', mmap=True); print('ckpt ok')" "$CK" 2>/dev/null \
  || /home/user/miniconda3/envs/starVLA/bin/python -c "import torch,sys; torch.load(sys.argv[1], map_location='cpu', mmap=True); print('ckpt ok')" "$CK"
echo "[$(date +%T)] launching progress-shift run on GPUs 0-3"
STARVLA_CUDA_DECODER_CACHE=150 bash examples/realRobots/H1_2_car_real/train_files/run_car_yaml.sh "$YAML" 0,1,2,3 29763
echo "[$(date +%T)] run finished"
