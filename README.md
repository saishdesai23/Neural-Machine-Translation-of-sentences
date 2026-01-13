# 🧠 Neural Machine Translation (NMT) System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-v1.x-ee4c2c?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Active-success?style=flat)]()

## 📖 Overview

This project presents a comprehensive study and implementation of Neural Machine Translation (NMT) systems for translating English sentences into French. We explore the evolution of sequence modeling by implementing and comparing two distinct deep learning architectures: **Recurrent Neural Networks (RNNs)** and **Transformers**.

The repository serves as both a research platform and a technical guide, demonstrating the shift from sequential processing to attention-based mechanisms in Natural Language Processing.

---

## 🏗 architectures Implemented

We have implemented two primary translation pipelines:

### 1. RNN with Attention (Seq2Seq)
A classic Encoder-Decoder framework enhanced with Bahdanau Attention. This architecture processes sequences step-by-step, maintaining a hidden state to capture context.

<p align="center">
  <img width="50%" src="https://machinelearningmastery.com/wp-content/uploads/2021/09/bahdanau_1.png" alt="RNN with Attention">
  <br>
  <em>Figure 1: RNN Encoder-Decoder with Attention Mechanism</em>
</p>

### 2. Transformer (Self-Attention)
The state-of-the-art architecture based on the "Attention Is All You Need" paper. It eliminates recurrence in favor of Multi-Head Self-Attention, allowing for parallelization and better handling of long-range dependencies.

<p align="center">
  <img width="50%" src="https://miro.medium.com/max/856/1*ZCFSvkKtppgew3cc7BIaug.png" alt="Transformer Architecture">
  <br>
  <em>Figure 2: The Transformer Architecture</em>
</p>

---

## 📂 Project Structure

The codebase is modularized for clarity and reusability:

- **Notebooks**:
  - `Neural_Machine_Translation_RNN.ipynb`: Complete research pipeline for the RNN model (Preprocessing → Training → Eval).
  - `Neural_Machine_Translation_Transformers.ipynb`: Complete research pipeline for the Transformer model.
  
- **Core Modules**:
  - `rnnencoder.py` / `rnndecoder.py`: PyTorch definitions for RNN components.
  - `transformerencoder.py` / `transformerdecoder.py`: PyTorch definitions for Transformer components.
  - `positionalembedding.py`: Implementation of positional encoding for Transformers.
  - `decodingalgorithm.py`: Inference logic (Greedy decoding / Beam search).

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.x
*   PyTorch
*   Pandas, NumPy, NLTK
*   Jupyter Notebook

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/saishdesai23/Neural-Machine-Translation-of-sentences.git
    cd Neural-Machine-Translation-of-sentences
    ```

2.  **Dataset**
    The notebooks are configured to automatically download the English-French dataset from `manythings.org`. Ensure you have an internet connection for the first run.

3.  **Run the Analysis**
    Launch Jupyter Notebook and open either of the main architecture files:
    ```bash
    jupyter notebook Neural_Machine_Translation_RNN.ipynb
    # OR
    jupyter notebook Neural_Machine_Translation_Transformers.ipynb
    ```

---

## 📊 Evaluation

The models are rigorously evaluated using **BLEU (Bilingual Evaluation Understudy)** scores. 
*   **BLEU-1 to BLEU-4** metrics are calculated to assess precision across n-grams (unigrams to 4-grams).
*   Comparative analysis is performed on an unseen test set to validate generalization capabilities.

---

## 📜 License
This project is open-source and available for educational and research purposes.
