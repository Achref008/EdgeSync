# Robust Decentralized Federated Learning (DFL) for Heterogeneous Edge Devices

This project implements a decentralized federated learning (DFL) framework across heterogeneous AI nodes, specifically NVIDIA Jetson (CNN-based) and BrainChip Akida (neuromorphic SNN/CNN). It uses Metropolis-Hastings-style consensus, adaptive momentum updates, and cross-architecture weight projection.

## Features
- Metropolis-Hastings gossip averaging
- Jetson ↔ Akida model projection
- Peer scoring and dropout tolerance
- Dynamic topology and fault resilience
- Cross-entropy + distillation for heterogeneous training

## Target Devices
-  NVIDIA Jetson Nano/Xavier/Orin (PyTorch-compatible)
- BrainChip Akida neuromorphic boards (via projected SNN/CNN)
- Simulated CPU nodes for rapid testing

## Structure
- `jetson_node/`: CNN models, projection, and comms for Jetson
- `akida_node/`: Placeholder for Akida integration
- `docs/`: Contains device compatibility explanation (PDF)

## Getting Started
```bash
git clone https://github.com/YOUR_USERNAME/DFL-Heterogeneous.git
cd jetson_node
pip install -r requirements.txt
python main.py
