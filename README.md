# StreamingAI-Prototype — Decentralized Federated Learning for Heterogeneous Edge Devices (Jetson ↔ Akida)

This repository implements a **Decentralized Federated Learning (DFL)** prototype for **collaborative image classification** on heterogeneous edge hardware:

- **NVIDIA Jetson** nodes train CNN models locally (PyTorch)
- **BrainChip Akida** nodes participate as heterogeneous peers (SNN/CNN pipeline)

There is **no central server**.  
Each node trains on its own local image data and periodically **exchanges raw model parameters** with neighboring nodes to improve both **local performance (accuracy/loss)** and a **shared global model**.

---

## Why this project?

Real-world edge AI deployments often face:

- Distributed devices with **private or local-only data**
- **Unstable networks** (timeouts, dropouts, intermittent connectivity)
- **Heterogeneous hardware and model types** (CNN vs. SNN)

This project demonstrates how decentralized, peer-to-peer learning can continue **without a parameter server**, using gossip-based coordination, fault tolerance, and cross-architecture model alignment.

---

## How it works (one learning loop)

Each node repeatedly executes the following steps:

1. **Load local image dataset**
2. **Train locally** for a short number of epochs/steps
3. **Evaluate** locally (optional)
4. **Compute parameter updates (deltas)** relative to the previous round
5. **Serialize & send** updates to neighboring nodes via P2P TCP sockets
6. **Receive** updates from peers
7. **Project / align parameters** if communicating across CNN ↔ SNN architectures
8. **Aggregate updates** using Metropolis-Hastings gossip weights
9. **Apply momentum-stabilized update** to the local model
10. Continue training with improved weights

**Result:** all nodes predict images locally and **learn together**, while keeping raw data private.

---

## Key features

- **Serverless training** (true peer-to-peer decentralized learning)
- **Metropolis-Hastings gossip averaging** for consensus
- **Heterogeneous parameter exchange** (Jetson ↔ Akida via projection)
- **Momentum-stabilized updates** for convergence under instability
- **Dropout & timeout resilience** (nodes continue learning if peers disconnect)
- **Non-IID data support** (Dirichlet-based data splits)
- **Edge-ready implementation** (runs on real devices, not only simulations)

---

## Hardware & node roles

| Node Type | Count | Dataset | Role |
|----------|-------|--------|------|
| Jetson   | 4     | CIFAR-10 | Main CNN training nodes |
| Akida    | 1     | MNIST (+ optional CIFAR) | Lightweight heterogeneous learner |

All nodes **predict images locally** and **collaborate by sharing parameter updates only**.

---

## Repository structure

```text
StreamingAI-Prototype/
│
├── jetson_nodes/
│   Jetson-side implementation:
│   - local CNN training loop
│   - parameter exchange & aggregation
│   - peer-to-peer communication
│
├── akida_nodes/
│   Akida-side integration:
│   - SNN/CNN handling
│   - parameter projection/mapping
│   - placeholders for Akida SDK logic
│
├── config.py
│   Network topology, node roles, hyperparameters
│
├── requirements.txt
│   Python dependencies
│
├── Images/
│   Testbed photos and system diagrams
│
└── README.md
```

---

## Target Devices
- NVIDIA Jetson Nano/Xavier/Orin (PyTorch-compatible)
- BrainChip Akida neuromorphic boards (via projected SNN/CNN)
- Simulated CPU nodes for rapid testing

## Structure
- `jetson_nodes/`: CNN models, projection, and comms for Jetson
- `akida_nodes/`: Placeholder for Akida integration
- `docs/`: Contains device compatibility explanation (PDF)

## Getting Started
```bash
git clone https://github.com/Achref008/StreamingAI-Prototype.git
cd StreamingAI-Prototype

pip install -r requirements.txt

# Run a Jetson node
cd jetson_nodes
python main.py

# Run the Akida node
cd ../akida_nodes
python main.py
```

![Testbed](https://github.com/Achref008/StreamingAI-Prototype/blob/main/Images/Real%20image%20of%20the%20Testbed.PNG?raw=true)  
![Testbed2](https://github.com/Achref008/StreamingAI-Prototype/blob/main/Images/Diagram%20of%20the%20Testbed.PNG?raw=true)
