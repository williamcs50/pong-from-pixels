# Pong from Pixels

Pong-from-Pixels is a ground-up Deep Q-Network (DQN) project that learns to play Atari Pong directly from raw screen pixels on local GPU hardware, documenting the real training curve, including failures, as a reproducible, portfolio-grade reinforcement learning case study.

## Setup

```bash
git clone https://github.com/williamcs50/pong-from-pixels.git
cd pong-from-pixels
pip install -r requirements.txt
```

## Verify your environment

```bash
python scripts/check_environment.py
```

Confirms Gymnasium + Pong, PyTorch version, CUDA availability, GPU name, and a live matrix multiply on the GPU.

## Run the random agent

```bash
python src/random_agent.py
```
