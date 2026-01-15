# Experimental Results

## 1. Loss under Disconnections (5 Heterogeneous Nodes)

![Loss](https://github.com/Achref008/StreamingAI-Prototype/blob/main/Images/5nodes%20loss%20average%20under%20Disconnections.PNG)  
This figure shows the average training loss across a decentralized network under intermittent connectivity.
Orange dashed lines indicate node disconnections, while green dashed lines indicate reconnections.

**Parameters:**
- Dataset: CIFAR-10 (Jetsons) and MNIST (Akida) 
- Nodes: 4 NVIDIA Jetson (CNN) nodes + 1 BrainChip Akida (SNN/CNN) node
- Momentum buffer: β = 0.9
- Learning rate schedule: Cosine annealing
- Gradient clipping: 5
- Local training steps: TAU1 = 20 batches per round

This experiment demonstrates that momentum buffering stabilizes convergence despite network volatility and packet loss.

---

## 2. CIFAR-10 Validation Accuracy vs Momentum β


![Accuracy](https://github.com/Achref008/StreamingAI-Prototype/blob/main/Images/accuracy.PNG) 

This figure compares validation accuracy on unseen CIFAR-10 data for different momentum coefficients β.

**Parameters:**
- Dataset: CIFAR-10
- Nodes: 4 NVIDIA Jetson (CNN) nodes + 1 BrainChip Akida (SNN/CNN) node
- Decentralized gossip-based learning
- Same learning rate, batch size, and topology of los.png image for all runs.

Lower β converges faster initially but shows higher variance, while higher β (β = 0.9) provides smoother and more stable convergence.

---

## 3. CIFAR10–MNIST Training Loss (Non-IID)

![Loss NonIID](https://github.com/Achref008/StreamingAI-Prototype/blob/main/Images/loss.PNG) 

The raining loss evolution under heterogeneous non-IID data distribution across CNN and neuromorphic nodes is shown in this figure.

**Parameters:**
- Dataset: CIFAR-10
- Nodes: 4 NVIDIA Jetson (CNN) nodes + 1 BrainChip Akida (SNN/CNN) node
- Learning rate: 3e-4  
- Batch size: 128  
- Dirichlet non-IID factor: α = 0.1  
- Local steps per round: TAU1 = 50  
- Optimizer: Adam

Higher momentum significantly improves stability and reduces oscillations in cross-architecture decentralized learning.
