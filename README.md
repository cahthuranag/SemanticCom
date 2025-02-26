# Semantic Communication System Simulation Model (AI-RAN)

## Introduction
This simulation model demonstrates **semantic-aware transmission** in wireless networks using a **deep learning architecture**. The system employs **jointly trained CNN-based transmitters and receivers** to achieve end-to-end optimization of semantic communication.

## System Components

### 1. Transmitter Design
- Utilises multiple **CNN layers** for **semantic feature extraction**.
- Implements **Generalized Divisive Normalization (GDN)** after convolutional layers.
- Processes input data to **preserve essential semantic information** while **removing redundancy**.
- Final layers **map extracted features to channel symbols** for efficient transmission.

### 2. Wireless Channel Model
- Simulates **AWGN and Rayleigh fading channels**.
- Incorporates **path loss** and **multi-path effects**.
- Features **adjustable SNR levels** and **interference patterns** to test system performance.

### 3. Receiver Architecture
- Utilises **CNN layers** for **signal processing**.
- Focuses on **preserving semantic meaning** rather than exact signal reproduction.
- Achieves more **efficient communication** compared to traditional **bit-level accuracy systems**.

## Training Implementation
- Trained using the **CIFAR-10 dataset**.
- **End-to-end optimization** for balancing **semantic accuracy** and **reconstruction quality**.
- The loss function ensures **semantic fidelity** while adapting to **channel conditions**.

## Simulation Results
- **PSNR Comparison**: The received image quality is evaluated under **deep learning-based wireless communication** (joint source-channel coding) versus **traditional models** (LDPC for channel coding, BPG for source coding) at different SNR levels.
- **Image Classification Accuracy**: The system's ability to classify images at the destination is measured under different SNR conditions.

