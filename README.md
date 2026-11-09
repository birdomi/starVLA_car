# starVLA × RMBench

[RMBench](https://github.com/RoboTwin-Platform/RMBench)([arXiv 2603.01229](https://arxiv.org/abs/2603.01229))는 RoboTwin 2.0(SAPIEN) 위에 만든 **메모리 의존 양팔 조작 벤치마크**입니다. 이 repo는 [starVLA](https://github.com/starVLA/starVLA) 기반이고, 이 README는 RMBench로 다음 일을 하는 방법을 처음부터 순서대로 설명합니다.

- **Qwen3.5-2B + EpisodeMemory + DiT** 모델(`QwenGR00T_EpisodeMemory`, end-to-end) 학습
- 시뮬레이터 평가
- 롤아웃 영상 만들기

> 원래 starVLA README(프레임워크 소개, LIBERO 등 다른 벤치마크)는 upstream repo에 있습니다. 이 repo의 git history에서도 볼 수 있습니다: `git show 0ed0aad:README.md`.

---

## 목차

1. [전체 구조](#1-전체-구조)
2. [경로와 파일](#2-경로와-파일)
3. [설치](#3-설치)
4. [데이터 변환](#4-데이터-변환)
5. [학습](#5-학습)
6. [평가](#6-평가)
7. [롤아웃 영상](#7-롤아웃-영상)
8. [모델 구조](#8-모델-구조)
9. [지금까지 결과](#9-지금까지-결과)
10. [문제 해결](#10-문제-해결)

---

## 1. 전체 구조

```
[학습]
RMBench demo_clean (HDF5, 9 tasks × 50 eps)
   └─ convert_rmbench_to_lerobot.py ─▶ LeRobot v2.0 (playground/Datasets/RMBench)
                                          └─ train_starvla.py (accelerate + DeepSpeed ZeRO-2, 4 GPU)
                                               └─ playground/Checkpoints/<run_id>/

[평가]  에피소드마다 3개 프로세스가 한 줄로 연결됩니다.
RMBench 시뮬레이터 client        XPolicyLab policy server            starVLA websocket server
(RMBench env, GPU, 렌더링)  ──▶  + starVLA_dev adapter       ──▶   (starVLA env, GPU, 모델 추론)
                                 (RMBench env, CPU, 형식 변환)       deployment/model_server/server_policy.py
```

conda 환경은 두 개를 씁니다.

| 환경 | 용도 | 주요 버전 |
|---|---|---|
| `starVLA` | 데이터 변환, 학습, 정책 서버 | python 3.10, torch 2.12 cu130, transformers 5.5.0, flash_attn 2.8.3, flash-linear-attention 0.5.2, deepspeed 0.16.9 |
| `RMBench` | 시뮬레이터, XPolicyLab | python 3.10, torch 2.7.1 cu128, sapien 3.0.0b1, mplib 0.2.1, curobo 0.7.8 |

GPU는 RTX PRO 6000 Blackwell(97.9 GB, sm_120) 기준입니다.

---

## 2. 경로와 파일

이 서버 기준 경로입니다. 다른 곳에서 쓰려면 아래 표의 환경변수로 바꾸세요.

| 무엇 | 경로 | 바꾸는 법 |
|---|---|---|
| starVLA repo | `/mnt/DATA2/yyg/starVLA` | `STARVLA_ROOT` (adapter) |
| RMBench repo | `/mnt/DATA2/yyg/RMBench` | `RMBENCH_ROOT` |
| RMBench conda env 이름 | `RMBench` | `RMBENCH_ENV` |
| starVLA python | `~/miniconda3/envs/starVLA/bin/python` | `STARVLA_PYTHON` |
| 변환된 데이터 | `/mnt/DATA2/yyg/RMBench/lerobot/<task>` (`playground/Datasets/RMBench`로 symlink) | yaml `data_root_dir` |
| Qwen3.5-2B | `playground/Pretrained_models/Qwen3.5-2B` (HF `Qwen/Qwen3.5-2B`) | yaml `base_vlm` |
| 메모리 사전학습 ckpt | `playground/Checkpoints/memory_pretrain_multivlm_n32d8_t1024_egodex/checkpoints/` | yaml `episode_memory.init_from` |
| 평가 영상 | `/mnt/DATA2/yyg/RMBench/eval_result/<task>/starVLA_dev/demo_clean/<ckpt_tag>/<timestamp>/episode0.mp4` | — |

repo 안의 RMBench 관련 파일은 모두 `examples/simBenchmarks/RMBench/` 아래에 있습니다.

```
examples/simBenchmarks/RMBench/
├── train_files/
│   ├── convert_rmbench_to_lerobot.py        # HDF5 → LeRobot v2.0 변환
│   ├── data_registry/data_config.py         # mixture: rmbench_9, rmbench_9_h50, rmbench_m1, rmbench_mn, rmbench_<task> ...
│   ├── starvla_qwen35_2b_gr00t_dit_episodememory_e2e_rmbench9.yaml        # 30k 레시피 (chunk 16)
│   ├── starvla_qwen35_2b_gr00t_dit_episodememory_e2e_rmbench9_100k_b96.yaml # 100k 레시피 (chunk 50, random_phase)
│   └── starvla_qwen35_gr00t_dit_v2_rmbench.yaml # 메모리 없는 DiT v2 기준선 (아직 학습 안 함)
└── eval_files/
    ├── eval_rmbench_parallel.py             # 병렬 평가 드라이버 (권장)
    ├── eval_rmbench.sh                      # 태스크를 하나씩 도는 단순 평가
    ├── rmbench_local.patch                  # RMBench repo에 적용할 로컬 패치 (필수)
    └── xpolicylab/starVLA_dev/              # XPolicyLab adapter (RMBench 관측 ↔ starVLA 서버)
```

이 모델을 학습할 때 쓰는 공용 코드는 아래와 같습니다.

- `starVLA/model/framework/VLM4A/QwenGR00T_EpisodeMemory.py`: 모델
- `starVLA/dataloader/gr00t_lerobot/datasets.py`: 에피소드 stream loader, `random_phase`
- `deployment/model_server/policy_wrapper.py`: 서버 쪽 에피소드별 메모리 상태
- `examples/simBenchmarks/Robotwin/train_files/data_registry/data_config.py`: robot type `robotwin`, `robotwin_h50`
- `examples/simBenchmarks/launch_sim_train_local.sh`: 학습 런처

---

## 3. 설치

### 3.1 starVLA 환경

```bash
conda create -n starVLA python=3.10 -y
conda activate starVLA
cd /mnt/DATA2/yyg/starVLA
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install flash-linear-attention        # Qwen3.5 linear attention. 없으면 24층 중 18층이 느린 torch 경로로 감 (VLM forward 약 5배 느림)
pip install -e .
```

> **주의**: 이 서버의 `starVLA` env에는 예전 checkout(`/mnt/DATA1/yyg/starVLA`)이 editable로 설치돼 있습니다. 스크립트를 직접 python으로 실행할 때는 반드시 `export PYTHONPATH=/mnt/DATA2/yyg/starVLA`를 먼저 하세요. 이 README의 런처와 서버 스크립트는 PYTHONPATH를 알아서 설정합니다.

### 3.2 RMBench 환경과 데이터

```bash
git clone --recurse-submodules https://github.com/RoboTwin-Platform/RMBench.git /mnt/DATA2/yyg/RMBench
cd /mnt/DATA2/yyg/RMBench
```

conda 환경은 RMBench의 `scripts/_install.sh`로 만들지 **마세요**. 이 스크립트는 torch 2.4.1을 고정하는데, 이 버전에는 Blackwell(sm_120) 커널이 없습니다. 대신 이미 동작하는 RoboTwin 환경을 복제합니다.

```bash
conda create -n RMBench --clone RoboTwin     # torch 2.7.1+cu128, sapien 3.0.0b1, mplib 0.2.1(패치 적용됨), curobo 0.7.8
conda activate RMBench
pip install -e XPolicyLab --no-deps
pip install msgpack-numpy --index-url https://pypi.org/simple
```

RoboTwin 환경이 없다면 RoboTwin 2.0 설치 문서를 따르되, torch 2.7.1+cu128을 설치하고 curobo를 sm_120으로 다시 빌드해야 합니다. pytorch3d는 ABI가 맞지 않아 깨져 있는데, point cloud FPS에만 쓰이고 demo_clean에서는 꺼져 있으므로 그대로 두면 됩니다.

에셋과 데이터를 받습니다. 에셋은 1.3 GB, 데이터는 10 tasks × 50 eps로 22 GB입니다.

```bash
bash scripts/_download_assets.sh
bash scripts/_download_data.sh      # → data/demo_clean/<task>/aloha_agilex/
```

### 3.3 RMBench 로컬 패치와 adapter 연결

```bash
cd /mnt/DATA2/yyg/RMBench
git apply /mnt/DATA2/yyg/starVLA/examples/simBenchmarks/RMBench/eval_files/rmbench_local.patch
ln -s /mnt/DATA2/yyg/starVLA/examples/simBenchmarks/RMBench/eval_files/xpolicylab/starVLA_dev XPolicyLab/policy/starVLA_dev
```

패치는 `envs/_base_task.py`와 `scripts/eval_policy_xpolicylab.py`를 고치며, 아래 환경변수를 추가합니다. 원래 동작은 바뀌지 않고, 변수를 설정했을 때만 효과가 있습니다.

| 환경변수 | 기능 |
|---|---|
| `ROBOTWIN_EVAL_SHARD=k/K` | seed shard: `st_seed + k (mod K)`인 seed만 실행. 병렬 평가에 필요 |
| `ROBOTWIN_EVAL_START_SEED` | 죽은 shard를 마지막으로 끝난 seed 다음부터 다시 시작 |
| `ROBOTWIN_EVAL_VIDEO_GRID=1` | 평가 영상을 2×2 격자(third view │ head / left wrist │ right wrist, 640×480)로 저장. 기본 영상은 third view만 있어서 observe_and_pickup처럼 화면이 테이블을 가리는 태스크는 보기 어렵습니다 |
| `ROBOTWIN_SHADER_DIR`, `ROBOTWIN_RT_SPP`, `ROBOTWIN_RT_DEPTH`, `ROBOTWIN_RT_DENOISER` | 렌더러 설정(기본값 = 원본 `rt`/32/8/`oidn`) |

패치는 또 로그에 `[episode-start] seed=…`, `[episode] seed=… success=… steps=…` 줄을 찍습니다. 병렬 드라이버는 이 줄로 진행 상황을 추적합니다.

---

## 4. 데이터 변환

`starVLA` env에서 실행합니다. 변환기는 RMBench repo의 XPolicyLab `decode_image_bit`를 import하므로 `RMBENCH_ROOT`가 맞아야 합니다.

```bash
cd /mnt/DATA2/yyg/starVLA
python examples/simBenchmarks/RMBench/train_files/convert_rmbench_to_lerobot.py --all --out_root /mnt/DATA2/yyg/RMBench/lerobot --workers 16
ln -s /mnt/DATA2/yyg/RMBench/lerobot playground/Datasets/RMBench
# 태스크 하나만: --task swap_blocks --out /mnt/DATA2/yyg/RMBench/lerobot/swap_blocks   (디버그: --limit 2)
```

- 결과물은 10 tasks, 330,541 frames, 1.9 GB입니다. 카메라 3대(`cam_high`, `cam_left_wrist`, `cam_right_wrist`, 320×240)와 15 Hz 데이터를 h264 mp4로 저장합니다.
- state와 action은 RoboTwin과 같은 14-D 절대 관절값 `[left arm 6, left gripper, right arm 6, right gripper]`입니다. HDF5에서 `action[t] == state[t+1]`이라서 모든 프레임을 그대로 씁니다.
- instruction은 에피소드마다 `seen` 목록에서 하나를 뽑습니다(seed 고정).
- 태스크는 paper의 9개에 `place_block_mat`이 더해진 10개입니다. paper 표에 있는 9개는 아래와 같습니다.
  - **M(1)**(과거 관측 한 번이 필요): `observe_and_pickup`, `rearrange_blocks`, `put_back_block`, `swap_blocks`, `swap_T`
  - **M(n)**(여러 번이 필요): `battery_try`, `blocks_ranking_try`, `cover_blocks`, `press_button`

mixture는 `train_files/data_registry/data_config.py`에 정의돼 있고, `examples/**/train_files/data_registry/`를 자동으로 스캔해 등록됩니다.

| mixture | 내용 |
|---|---|
| `rmbench_9` | paper 9 tasks, robot type `robotwin` (action chunk 16) |
| `rmbench_9_h50` | 같은 9 tasks, robot type `robotwin_h50` (action chunk 50) |
| `rmbench_all` / `rmbench_m1` / `rmbench_mn` / `rmbench_<task>` | 10 tasks / M(1) / M(n) / 태스크 하나 |

---

## 5. 학습

### 5.1 필요한 가중치

- `playground/Pretrained_models/Qwen3.5-2B`: HF `Qwen/Qwen3.5-2B` 원본
- `playground/Checkpoints/memory_pretrain_multivlm_n32d8_t1024_egodex/checkpoints/steps_50000_memory_qwen35_2b.pt`: 메모리 모듈
- `playground/Checkpoints/memory_pretrain_multivlm_n32d8_t1024_egodex/checkpoints/steps_50000.pt`: progress decision transformer

두 메모리 ckpt는 EgoDex로 메모리를 사전학습한 결과입니다(`starVLA/training/train_memory_pretrain.py`, KISTI에서 학습). 다른 서버에서 쓰려면 이 디렉터리를 복사하세요.

### 5.2 실행

```bash
cd /mnt/DATA2/yyg/starVLA
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash examples/simBenchmarks/launch_sim_train_local.sh \
    examples/simBenchmarks/RMBench/train_files/starvla_qwen35_2b_gr00t_dit_episodememory_e2e_rmbench9_100k_b96.yaml \
    4,5,6,7 29660 \
    --datasets.vla_data.per_device_batch_size=24      # 뒤에 --key=value로 yaml 값을 덮어쓸 수 있음
```

- 런처는 `setsid nohup`으로 백그라운드에서 띄우고 PID와 로그 경로를 출력합니다.
- 인자 순서는 `<yaml> [GPU 목록] [master port] [덮어쓸 값 ...]`입니다.
- 로그는 repo 루트의 `sim_train_<yaml 이름>_<MMDD_HHMM>.log`에 남습니다.
- 체크포인트는 `playground/Checkpoints/<run_id>/checkpoints/steps_<N>_pytorch_model.pt`(약 5.6 GB)에 저장되고, 학습이 끝나면 `final_model/pytorch_model.pt`가 생깁니다. `config.yaml`, `config.full.yaml`, `dataset_statistics.json`은 같은 run 디렉터리에 있고, 평가 때 adapter와 서버가 이 파일을 읽습니다.
- wandb 설정은 yaml의 `wandb_entity` / `wandb_project`에 있습니다. 본인 계정으로 바꾸거나 `WANDB_MODE=offline`을 쓰세요.

진행 상황 확인:

```bash
LOG=$(ls -t sim_train_*rmbench9*.log | head -1)
tr '\r' '\n' < $LOG | grep -aoE '[0-9]+/100000 \[[^]]*\]' | tail -1      # 진행 step, 남은 시간
tr '\r' '\n' < $LOG | grep -a -A14 'Step [0-9]*, Loss' | tail -15       # action_dit_loss, memory/progress_ce 등
```

### 5.3 두 레시피

| 설정 | 30k (`..._rmbench9.yaml`) | 100k (`..._rmbench9_100k_b96.yaml`) |
|---|---|---|
| data_mix | `rmbench_9` | `rmbench_9_h50` |
| action_horizon | 16 | 50 (평가 때는 앞 16개만 실행) |
| batch | 28/GPU × 4 = 112 | 24/GPU × 4 = 96 |
| stream.random_phase | false (프레임 0, 16, 32 …만 학습) | true (시작 프레임을 [0, 16)에서 무작위로 → 모든 프레임 학습) |
| max_train_steps / save_interval | 30k / 5k | 100k / 20k |
| 속도 / 소요 시간 | 1.94 s/step, 약 16 h | 1.82 s/step, 약 50 h |
| GPU 메모리 peak | 약 91 GB | 83.7 GB |

두 레시피의 공통 설정:
- LR: VLM 1e-5, DiT/메모리/decision 1e-4
- 스케줄러: cosine_with_min_lr(최소 1%), warmup 1000 step
- 옵티마이저: AdamW (0.9, 0.95)
- grad clip 1.0
- `skip_nonfinite_steps`

**배치 크기 참고**(97.9 GB GPU)
- chunk 16에서 56/GPU는 첫 VLM forward에서 OOM이 납니다.
- 32/GPU는 돌기는 하지만 97.2 GB로 한계에 걸리고 3.5 s/step으로 느립니다.
- activation checkpointing은 쓰지 않습니다.

> **체크포인트에는 모델 가중치만 들어 있고 optimizer 상태가 없습니다.** 중간에 멈춘 학습은 이어서 돌릴 수 없으니, `save_interval`과 `max_train_steps`를 처음에 잘 정하세요.

---

## 6. 평가

평가 프로토콜은 RMBench paper를 따릅니다.
- 태스크마다 100 에피소드를 돌리고, sim seed는 `100000 × (1 + seed)`부터 씁니다.
- instruction은 `seen`입니다.
- step 상한은 `env_cfg/task_config/_eval_step_limit.yml`에 있습니다. 예: swap_blocks 1000, blocks_ranking_try 3500, observe_and_pickup 250.
- 최종 점수는 9 tasks 평균 성공률입니다.
- expert check에 실패한 seed(불안정한 장면)는 건너뛰고 다음 seed로 채우는데, 이것은 RMBench 원래 동작입니다.

평가는 오래 걸리므로 `setsid nohup ... &`로 띄워서 터미널과 분리하세요.

### 6.1 빠른 확인: 한 태스크, 4 에피소드, 약 5분

```bash
cd /mnt/DATA2/yyg/starVLA
CK=playground/Checkpoints/starvla_qwen35_2b_dit_episodememory_e2e_egodex50k_rmbench9_100k_b96/checkpoints/steps_40000_pytorch_model.pt
OUT=results/rmbench/quick_40k
mkdir -p $OUT
setsid nohup $HOME/miniconda3/envs/starVLA/bin/python examples/simBenchmarks/RMBench/eval_files/eval_rmbench_parallel.py $CK \
    --tasks swap_blocks --gpus 4,5,6,7 --slots_per_gpu 1 --shards 4 --test_num 4 --log_dir $OUT \
    > $OUT/driver.log 2>&1 < /dev/null &
```

GPU 4개에 shard 4개를 하나씩 올려서 seed 100008, 100001, 100002, 100015로 1 에피소드씩 돌립니다(expert check에 실패한 seed는 건너뜀). 1000-step 에피소드 하나에 약 3분 걸립니다.

### 6.2 전체 평가: 9 tasks × 100 에피소드

```bash
OUT=results/rmbench/<run>_<ckpt>_parallel     # 생략하면 이 기본 경로를 씀
setsid nohup $HOME/miniconda3/envs/starVLA/bin/python examples/simBenchmarks/RMBench/eval_files/eval_rmbench_parallel.py $CK \
    --tasks 9 --gpus 4,5,6,7 --slots_per_gpu 5 --shards 10 --test_num 100 \
    > results/rmbench/eval_driver.log 2>&1 < /dev/null &
```

드라이버가 하는 일:

1. GPU마다 starVLA 정책 서버를 하나씩 띄웁니다(약 6 GB).
2. 각 태스크를 seed shard 10개로 나눠 XPolicyLab job으로 돌립니다(시뮬레이터 한 대당 약 7.5 GB).
3. 각 job은 자기 adapter를 쓰므로, 에피소드 메모리 slot이 서로 섞이지 않습니다.
4. step 상한이 긴 태스크부터 시작합니다.
5. **시뮬레이터가 멈추면 감지해서 kill한 뒤, 끝난 에피소드 다음부터 다시 실행합니다.**
   - 감지 조건: client 소켓이 CLOSE-WAIT로 5분 지속되거나, 로그와 영상이 20분 동안 늘지 않음
   - 같은 seed에서 두 번 죽으면 그 seed를 건너뛰고 `summary.txt`에 표시합니다.
6. 같은 `--log_dir`로 다시 실행하면 끝난 shard는 건너뜁니다. 중간에 끊겨도 같은 명령으로 이어서 돌리면 됩니다.

주요 옵션은 아래와 같습니다. 전체 목록은 `--help`로 볼 수 있습니다.

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--tasks` | `9` | `all` / `9` / `m1` / `mn` / 쉼표로 구분한 태스크 목록 |
| `--gpus` | `4,5,6,7` | 시뮬레이터 GPU |
| `--server_gpus` | (= gpus) | 정책 서버 GPU. 학습 중인 GPU와 나눠 쓸 때 지정 |
| `--slots_per_gpu` | 4 | GPU당 동시에 도는 시뮬레이터 수 |
| `--shards` | 10 | 태스크당 seed shard 수(`test_num`을 나눠떨어지게) |
| `--seed` | 0 | 평가 seed |
| `--no_video_grid` | — | 2×2 격자 대신 기본 third view 영상 |
| `--idle_min` / `--close_wait_min` / `--max_attempts` / `--skip_after` | 20 / 5 / 3 / 2 | hang 감지와 재시도 설정 |

- **학습과 GPU를 나눠 쓸 때**는 서버와 시뮬레이터를 다른 GPU에 두세요. 예: `--gpus 5,7 --server_gpus 4,6 --slots_per_gpu 1`.
  - 학습은 1.84 s/it에서 약 2.5 s/it으로 느려집니다.
  - 시뮬레이터 hang은 약 5배 늘어납니다([10절](#10-문제-해결)).
- **소요 시간 참고**: 시뮬레이터 step 하나에 약 0.2 s, job 시작에 약 1분이 듭니다. 9 tasks × 100을 4 GPU × 5 slot으로 돌리면 반나절쯤 걸립니다.
- **단순 순차 평가**가 필요하면 `bash examples/simBenchmarks/RMBench/eval_files/eval_rmbench.sh <ckpt> <tasks> [gpu] [test_num] [seed]`를 쓰세요. 태스크를 하나씩 돌리고 hang 감지는 없습니다.

### 6.3 결과 파일

```
<log_dir>/
├── summary.txt          # "swap_blocks: Final success rate: 1/4 = 25.0%" ... + "average over N tasks"
├── results.json         # shard별: 에피소드마다 seed / success / steps, skipped_seeds, attempts, minutes
├── driver.log           # 진행 상황, hang 감지/재시작 기록
├── server_gpu<g>.log    # 정책 서버 로그
└── jobs/<task>_s<k>.log # job(shard) 로그 (시뮬레이터 출력 전체)
```

### 6.4 adapter 설정 (`eval_files/xpolicylab/starVLA_dev/deploy.yml`)

모든 키는 대괄호 안의 환경변수로 덮어쓸 수 있습니다. 보통은 기본값(`auto`)을 그대로 쓰면 됩니다.

- `memory: auto`: run config에서 stream EpisodeMemory를 감지합니다. 감지되면 요청마다 `mem_slot`, `mem_t`, `mem_reset`을 보내고, 서버가 에피소드별로 메모리를 이어서 들고 갑니다.
- `execute_horizon: null`: 메모리 모델이면 학습 때의 stream stride(16)만큼 실행합니다. 그래서 정책 호출이 학습 때와 같은 0, 16, 32 … 프레임 격자에 떨어지고, 호출 한 번에 메모리가 한 번 갱신됩니다.
- `send_state: auto`: ckpt의 `include_state`를 따릅니다.
- `control_hz: 15`
- 관절 순서 변환: XPolicyLab은 `[left 6, left grip, right 6, right grip]` 순서이고 모델은 `[left 6, right 6, left grip, right grip]` 순서입니다. adapter가 양방향으로 바꿔 줍니다.
- EpisodeMemory ckpt이면 `ROBOTWIN_USE_BF16=0`이 자동으로 켜집니다. VLM은 config dtype(bf16)으로, 메모리와 DiT는 fp32로 학습 때와 똑같이 돌립니다.

> XPolicyLab의 batch 모드(`eval_batch: true`)는 adapter 하나를 여러 환경이 같이 써서 메모리 모델에서는 메모리가 섞입니다. 쓰지 마세요. 병렬화는 위 드라이버의 shard로 합니다.

---

## 7. 롤아웃 영상

에피소드 영상은 RMBench repo 아래에 job 실행 한 번마다 디렉터리 하나로 저장됩니다.

```
/mnt/DATA2/yyg/RMBench/eval_result/<task>/starVLA_dev/demo_clean/<ckpt_tag>/<YYYY-MM-DD_HH-MM-SS_xxxxxx>/
├── episode0.mp4, episode1.mp4, ...   # 2×2: third view | head / left wrist | right wrist, 640×480
└── _result.txt
```

`<ckpt_tag>`는 `<run 디렉터리 이름>_<ckpt 파일 이름(확장자 제외)>`입니다(예: `..._rmbench9_100k_b96_steps_40000_pytorch_model`).

**어떤 seed가 어느 영상인지 찾기**

- seed와 성공 여부: `results.json`에서 shard별 seed와 success를 봅니다.
- 그 shard의 영상 디렉터리: job 로그에서 찾습니다.

```bash
grep -a "Data has been saved" results/rmbench/quick_40k/jobs/swap_blocks_s2.log
# → eval_result/swap_blocks/starVLA_dev/demo_clean/<ckpt_tag>/2026-11-06_16-48-52_555009/_result.txt
```

shard가 재시작됐다면 시도마다 디렉터리가 따로 생깁니다. `episode<i>.mp4`의 i는 그 시도 안에서의 순번입니다.

**여러 에피소드를 이름표 붙여 2×2로 합치기**(ffmpeg)

길이가 다른 영상은 마지막 프레임을 복제해서 맞춥니다(`tpad`).

```bash
F=/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc          # 한글 이름표용
lab() { echo "drawtext=fontfile=$F:text='$1':x=8:y=8:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.7:boxborderw=6"; }
N=1000                                                          # 가장 긴 에피소드의 프레임 수 (보통 그 태스크의 step 상한)
pad="tpad=stop_mode=clone:stop=-1,trim=end_frame=$N"
ffmpeg -y -i s0.mp4 -i s1.mp4 -i s2.mp4 -i s3.mp4 -filter_complex \
  "[0]$pad,$(lab 'seed 100008 실패')[a];[1]$pad,$(lab 'seed 100001 실패')[b];[2]$pad,$(lab 'seed 100002 성공')[c];[3]$pad,$(lab 'seed 100015 실패')[d];[a][b][c][d]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[o]" \
  -map "[o]" -c:v libx264 -crf 30 -pix_fmt yuv420p grid.mp4        # 1280×960

# 같은 seed를 두 ckpt로 나란히 비교: [0]$pad[a];[1]$pad[b];[a][b]hstack=inputs=2[o]
```

- 이름표 문자열에 `:`를 넣으면 drawtext 옵션 구분자로 읽혀서 깨집니다. `-`나 `·`를 쓰세요.
- 파일이 크면 `-crf 30`에 `hqdn3d=3:3:4:4` 필터를 더하면 1/4 정도로 줄어듭니다. Notion에 올리기 좋은 크기입니다.

---

## 8. 모델 구조

`QwenGR00T_EpisodeMemory` (`condition: concat`, `stream.vlm_grad: true` = end-to-end)

```
현재 프레임 t (3 카메라 224×224) + instruction
        │
        ▼
  Qwen3.5-2B (원본 가중치, 학습함, lr 1e-5) ── last hidden states ──┐
        │                                                          │
        ▼ 16 프레임(1.07 s)마다 한 번                               ▼
  EpisodeMemory (32 tokens × d768, 8층, gated attention)  ──▶  [VLM tokens ‖ memory tokens]
   · 에피소드 프레임 0부터 순서대로 갱신 (stream)                   │ cross-attention
   · TBPTT: 마지막 4번의 갱신까지 역전파                             ▼
   · EgoDex 사전학습으로 초기화, core는 freeze                DiT-B (16층, flow matching)
   · progress head (decision transformer, CE × 0.1)                │
                                                                    ▼
                                                    14-D 절대 관절 action chunk (16 또는 50)
```

- 메모리 core(`init_tokens`, `in_proj`, `blocks`, `progress_head`, `progress_embed`, `decision.jdt`)는 freeze합니다.
- 학습하는 부분은 Qwen, DiT, 메모리의 token_resize / out_norm / mem_proj / source embedding, decision의 progress query와 projection입니다.
- DiT가 메모리를 읽는 projection은 0으로 초기화돼서, 학습 시작 시점에는 메모리가 없는 모델과 같습니다.
- 학습 loader는 에피소드를 처음부터 16 프레임 간격으로 걸어가는 stream을 배치로 묶습니다(`datasets.vla_data.stream`). `random_phase: true`이면 그 격자의 시작점이 [0, 16)에서 무작위로 정해집니다.

---

## 9. 지금까지 결과

(2026-11-06 기준. paper 수치: Mem-0 42.0%, Pi0.5 10.4%, X-VLA 9.8%, ACT 5.9%, DP 5.8%)

| ckpt | 평가 | 결과 |
|---|---|---|
| 30k (`..._rmbench9_30k_b112/final_model`) | 9 tasks × 100 | **2.7%** (24/900) |
| 100k 레시피 40k (`..._rmbench9_100k_b96/checkpoints/steps_40000`) | swap_blocks 4 eps | 1/4 (같은 seed에서 30k는 0/4. n=4라 의미 없음) |

30k의 태스크별 결과(/100): observe_and_pickup 4, rearrange_blocks 2, put_back_block 0, swap_blocks 8, swap_T 2, battery_try 8, blocks_ranking_try 0 (2 seeds skipped), cover_blocks 0, press_button 0.

**관찰**
- 30k는 학습 데이터에서 progress 정확도가 0.999까지 올랐지만 sim에서는 2.7%였습니다. `random_phase: false`라서 각 에피소드의 1/16 프레임(0, 16, 32 …)만 반복해서 본 것이 과적합 원인으로 의심됩니다(검증 안 됨). 100k 레시피는 이 점 때문에 `random_phase: true`로 바꿨습니다.
- 100k 학습은 47.5k step에서 멈췄고, `steps_20000`과 `steps_40000` ckpt만 남아 있습니다.
- 자주 보이는 실패 패턴(swap_blocks): 블록을 3번 옮겨야 하는데 2번만 옮기고 버튼을 누릅니다. 즉 "몇 번 옮겼는지"를 기억하지 못합니다.
  - swap_blocks 성공 조건: 블록이 바뀐 트레이에서 x 4 cm, y 6 cm 안에 있고 z < 0.765, 오른쪽 그리퍼가 열려 있으며 버튼을 정확히 1번 눌러야 함
- DiT의 flow noise에는 seed가 없어서, 같은 ckpt와 같은 seed로 돌려도 결과가 달라질 수 있습니다. 적은 에피소드로 비교할 때는 이 점을 주의하세요.

---

## 10. 문제 해결

| 증상 | 원인 / 대처 |
|---|---|
| 시뮬레이터가 에피소드 중간에 멈춤 (CPU 0%, client 소켓 CLOSE-WAIT) | SAPIEN 카메라 readback(Vulkan, ray tracing) hang입니다. GPU를 학습과 나눠 쓸 때 훨씬 잦습니다. 병렬 드라이버가 자동으로 kill하고 재시작합니다. 너무 잦으면 slot 수를 줄이거나, 학습과 GPU를 분리하거나, `ROBOTWIN_RT_DENOISER=none`을 쓰세요. hang은 seed와 상관없이 무작위로 일어납니다 |
| 첫 평가가 1시간 넘게 아무 출력이 없음 | `~/.cache/warp`가 비어 있으면 curobo 커널을 JIT 컴파일하는 중입니다(약 1 h, 처음 한 번만). 멈춘 게 아닙니다 |
| 로그에 OIDN `invalid handle`이 쏟아짐 | 정상입니다(denoiser가 원래 동작하지 않음). 무시하세요 |
| 평가 성공률이 이상하게 0%이고 서버 로그에 import 경로가 이상함 | 예전 editable 설치를 import한 경우입니다. `PYTHONPATH=/mnt/DATA2/yyg/starVLA`를 확인하세요 |
| 학습이 매우 느림 (VLM forward 5배) | `flash-linear-attention`이 없어서 Qwen3.5 linear attention이 torch fallback으로 돌고 있습니다 |
| 학습 OOM | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`를 켜고 batch를 줄이세요(5.3절 참고) |
| 학습을 이어서 하고 싶음 | ckpt에 optimizer 상태가 없어서 resume할 수 없습니다. 처음부터 다시 돌려야 합니다 |
| 평가 드라이버를 멈추고 싶음 | 드라이버 PID에 SIGTERM을 보내면 job들의 process group을 정리합니다(조금 걸림). 공용 서버이므로 `pkill` 대신 정확한 PID를 확인하고 kill하세요 |
| 메모리 모델 평가에서 에피소드끼리 결과가 섞이는 것 같음 | XPolicyLab batch 모드를 쓰지 말고, 드라이버의 shard(job마다 독립 adapter)를 쓰세요 |

---

## 참고

- RMBench: https://github.com/RoboTwin-Platform/RMBench, arXiv [2603.01229](https://arxiv.org/abs/2603.01229)
- starVLA (upstream): https://github.com/starVLA/starVLA, arXiv [2604.05014](https://arxiv.org/abs/2604.05014). MIT License ([LICENSE](LICENSE))
- RoboTwin 2.0: https://robotwin-platform.github.io
