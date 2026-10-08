## Project Description:

This project is based on a deep learning approach for detecting malicious network traffic. The system takes network traffic data, preprocesses it, and converts the traffic features into a suitable matrix format. It uses multi-scale feature extraction to capture patterns at different levels and then uses Transformer-based learning to understand the relationships between these features. Finally, the model classifies the traffic as either normal or an attack. The project uses Python and libraries such as PyTorch, NumPy, Pandas, Scikit-learn, and Matplotlib, along with techniques such as normalization, one-hot encoding, convolution, dilated convolution, Transformer encoders, multi-head self-attention, and focal loss.

## Understanding of the Base Research Paper — Novel-IDS

The base paper, **“Novel-IDS: A Novel Multi-Scale Network Intrusion Detection Model with Transformer,”** focuses on improving intrusion detection using deep learning. The main idea is to extract useful information from network traffic at different scales instead of depending on a single type of feature extraction. The model first preprocesses the network traffic data and converts it into a suitable form for deep learning.

The model uses multiple convolution branches to extract features at different scales. This helps the system capture both smaller patterns and larger patterns present in network traffic. After extracting these features, the model uses Transformer encoders to learn the relationships between different features. Multi-head self-attention allows the model to focus on important parts of the extracted traffic information.

Finally, the learned features are passed to the classification part of the model to identify whether the traffic is normal or malicious. The paper also uses focal loss to give more importance to difficult samples during training. From our study of the paper, we understood how multi-scale feature extraction and Transformer-based feature learning can be combined to build an intrusion detection system.

## Research Gaps We Identified:

## 1. **Multi-Class Classification and Generalization**  
   The system can be extended to handle multiple types of attacks instead of mainly focusing on normal versus attack traffic. We also want to study how well the model performs on different datasets and unseen attack types.

## 2. **Explainability**  
   The model can identify attacks, but understanding why a particular traffic sample was classified as an attack is still important. We aim to include explainability so that the important features or patterns behind a prediction can be understood.

## 3. **Moving Beyond Conventional Transformers**  
   The existing architecture uses conventional Transformer components. Instead of simply combining existing models, we aim to explore a new architecture built around attention-based principles, with our own feature-learning and attention mechanisms.

## Our Proposed Problem Description:

Our proposed work aims to develop a new network intrusion detection architecture based on the gaps identified from the Novel-IDS paper. Instead of directly combining conventional CNN, RNN, ANN, and Transformer models, we plan to design the architecture from the basic principles of attention-based feature learning. The goal is to create a model that can learn useful traffic patterns at different levels while maintaining the ability to capture relationships between network features.

The proposed system will focus on three main improvements: better multi-class attack classification and generalization, improved explainability of the model's decisions, and moving beyond the conventional Transformer architecture. The architecture will be designed and evaluated on network intrusion datasets to determine whether it can improve detection performance while also providing better understanding of the model's predictions.

Our current work focuses on studying the base architecture, identifying its limitations, designing the new architecture, and implementing and evaluating the proposed improvements. The final architecture and its individual components will be refined during the implementation and experimentation stages.
## Technologies and Concepts

### Programming and Libraries
- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

### Data Preprocessing
- Data Cleaning
- Feature Tokenization
- Feature Embedding
- One-Hot Encoding
- Min-Max Normalization

### Deep Learning and Model Concepts
- Attention Mechanism
- Multi-Head Attention
- Transformer Encoder
- Relational Attention
- Feature Interaction
- Residual Connections
- Layer Normalization
- Softmax Classification
- Focal Loss

### Intrusion Detection Concepts
- Network Intrusion Detection Systems (IDS)
- Binary Classification
- Multi-Class Classification
- Attack Pattern Detection
- Generalization to Different Network Traffic

  ## Base Paper Architecture — Novel-IDS

The Novel-IDS model follows a multi-scale feature extraction and Transformer-based learning approach. The overall architecture is shown below:

```text
┌─────────────────────────────┐
│     NETWORK TRAFFIC DATA    │
│                             │
│ NSL-KDD / CIC-DDoS2019 /    │
│ UNSW-NB15                   │
└──────────────┬──────────────┘
               │
               ▼
    ┌─────────────────────┐
    │    PREPROCESSING    │
    ├─────────────────────┤
    │ 1. One-hot encoding │
    │ 2. Outlier handling │
    │ 3. Min-Max normalize│
    │ 4. Matrixization    │
    └──────────┬──────────┘
               │
               ▼
         2-D Feature Matrix
               │
               ▼
      ┌──────────────────────────────┐
      │       MULTI-SCALE CNN        │
      │      FEATURE EXTRACTION      │
      └──────────────────────────────┘
               │
          ┌────┼────┐
          ▼    ▼    ▼
     ┌────────┐ ┌────────┐ ┌────────┐
     │BRANCH 1│ │BRANCH 2│ │BRANCH 3│
     │ 1×1    │ │ 1×1    │ │ 1×1    │
     │ Conv   │ │ Conv   │ │ Conv   │
     │        │ │   ↓    │ │   ↓    │
     │        │ │ 3×3    │ │5×5-like│
     │        │ │ Conv   │ │/dilated│
     └────┬───┘ └────┬───┘ └────┬───┘
          │          │          │
          └──────────┼──────────┘
                     │
                     ▼
           ┌─────────────────┐
           │       PwP       │
           │ Patching with   │
           │    Pooling      │
           └────────┬────────┘
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
    ┌───────────┐ ┌───────────┐ ┌───────────┐
    │Transformer│ │Transformer│ │Transformer│
    │ Backbone 1│ │ Backbone 2│ │ Backbone 3│
    └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
          │              │              │
          │       Each Transformer     │
          │       contains:            │
          │       • Positional Encoding│
          │       • Multi-Head         │
          │         Self-Attention     │
          │       • Layer Normalization│
          │       • Feed-Forward NN    │
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                ┌─────────────────┐
                │       CFE       │
                │ Cross Feature   │
                │   Enrichment    │
                └────────┬────────┘
                         │
                Features from all
                3 scales interact
                         │
                         ▼
                ┌─────────────────┐
                │  3 Linear Layers│
                │    Classifier   │
                └────────┬────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    FINAL OUTPUT     │
              │                     │
              │ Binary:             │
              │ Normal / Attack     │
              │                     │
              │ Multi-class:        │
              │ Attack category     │
              └─────────────────────┘

                    TRAINING
                       │
                       ▼
                 Focal Loss
                       +
                 Adam Optimizer
```

### Base Architecture Summary

The Novel-IDS architecture first preprocesses network traffic data and converts it into a 2-D feature matrix. A multi-scale CNN is then used to extract features at different scales. The extracted features are processed using patching with pooling and passed through three Transformer backbones. The Transformer components use positional encoding, multi-head self-attention, layer normalization, and feed-forward networks. The features from different scales are then combined using Cross Feature Enrichment (CFE) before being passed to the classifier. The model can be trained using focal loss with the Adam optimizer and can perform binary or multi-class classification.
 
 ## Proposed Architecture

Our proposed architecture is designed to learn relationships between network traffic features instead of directly relying on the conventional Transformer architecture used in the base paper.

The proposed architecture follows the flow below:

```text
Existing CSE Dataset
        ↓
Normalization
        ↓
Feature Tokenization
        ↓
Feature Embedding
        ↓
Relational Attention
        ↓
Feature Representation
        +
Cross-Feature Relationship Learning
        ↓
Relation-Aware Q / K / V
        ↓
Multi-Relation Heads
        ↓
Learning Different Types of Feature Relationships
        ↓
Relation Aggregation
        ↓
Residual Connection + Normalization
        ↓
Feature Interaction Block
        ↓
Learning Higher-Level Attack Patterns
        ↓
Residual Connection + Normalization
        ↓
Global Flow Representation
        ↓
Classification Head
        ↓
Linear Layer
        ↓
Softmax
        ↓
Binary Classification / Multi-Class Classification
```
Binary Classification / Multi-Class Classification
