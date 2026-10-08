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
