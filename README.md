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
- `jetson_nodes/`: CNN models, projection, and comms for Jetson
- `akida_nodes/`: Placeholder for Akida integration
- `docs/`: Contains device compatibility explanation (PDF)

## Getting Started
git clone https://github.com/Achref008/StreamingAI-Prototype.git
cd StreamingAI-Prototype

pip install -r requirements.txt

# Run a Jetson node
cd jetson_nodes
python main.py

# Run the Akida node
cd ../akida_nodes
python main.py

![Testbed](https://github.com/Achref008/StreamingAI-Prototype/blob/main/Images/Real%20image%20of%20the%20Testbed.PNG)

