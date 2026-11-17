## **Prerequisites**

To run this code successfully, the following software and libraries are required. The implementation is developed using Python and PyTorch.

### **1. Software Requirements**

- **Python:** Python3
- **Operating System:** Linux / Windows / macOS
- **Jupyter Notebook:** Required to execute the `.ipynb` files

### **2. Hardware Requirements**

- **GPU (Recommended):** NVIDIA GPU with CUDA support for faster training
- **CPU:** Supported, but training time will be significantly longer
- **RAM:** Minimum 8 GB recommended
- **Experimental Hardware:** NVIDIA Tesla P100 GPU

### **3. Required Python Libraries**

The following Python libraries are required:

- **PyTorch** (`torch`)
- **Torchvision**
- **NumPy**
- **Matplotlib**
- **scikit-learn**
- **Pandas**
- **tqdm**

Install the required dependencies using:

```bash
pip install torch torchvision numpy matplotlib scikit-learn pandas tqdm
```

---

## **Architecture**

The proposed architecture consists of the following major components:

1. DenseNet121 backbone
2. Hierarchical feature extraction
3. Feature Pyramid Network
4. Dual-Attention Module
5. Dual-Adaptive Attention
6. Feature aggregation
7. Fully connected classification head

**Figure:** Overall architecture of the proposed framework.

![Proposed Model Architecture](figures/proposed_model_arch.jpg)

---

## **Proposed Framework Mechanism**

The proposed framework processes breast ultrasound images through the following stages:

1. **Hierarchical Feature Extraction:** DenseNet121 extracts multilevel feature representations.
2. **Feature Projection:** Extracted features are projected into a common feature dimension.
3. **Feature Pyramid Construction:** FPN integrates multilevel representations through top-down feature propagation.
4. **Dual-Attention Refinement:** SE and CBAM independently refine the pyramid features.
5. **Adaptive Attention Fusion:** Learnable fusion weights combine the attention-enhanced representations.
6. **Feature Aggregation:** Refined pyramid features are pooled and concatenated.
7. **Classification:** The aggregated representation is passed through the classification head to predict the corresponding breast ultrasound class.

---

## **Experimental Setup**

| Parameter | Configuration |
|---|---|
| Backbone | DenseNet121 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Weight Decay | 0.0001 |
| Maximum Epochs | 100 |
| Early Stopping | Patience = 20 |
| Learning Rate Scheduler | ReduceLROnPlateau |
| Hardware | NVIDIA Tesla P100 GPU |
| Random Seeds | 12, 42, 99, 1111, and 1234 |

---

## Results

| Configuration | Params | BUS-BRA Acc. | BUS-BRA F1 | BUS-UCLM Acc. | BUS-UCLM F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| DenseNet121 | 7.6M | 0.8699 ± 0.0184 | 0.8683 ± 0.0187 | 0.8696 ± 0.0213 | 0.8707 ± 0.0205 |
| DenseNet + SE Attention | 7.7M | 0.8785 ± 0.0175 | 0.8798 ± 0.0172 | 0.8986 ± 0.0185 | 0.8991 ± 0.0189 |
| DenseNet + CBAM Attention | 7.7M | 0.8742 ± 0.0182 | 0.8741 ± 0.0193 | 0.8986 ± 0.0197 | 0.8991 ± 0.0212 |
| DenseNet + Dual-Attention | 8.0M | 0.8785 ± 0.0198 | 0.8793 ± 0.0205 | 0.8841 ± 0.0164 | 0.8840 ± 0.0165 |
| DenseNet + Adaptive Dual Fusion | 8.9M | 0.8849 ± 0.0172 | 0.8858 ± 0.0194 | 0.8841 ± 0.0148 | 0.8879 ± 0.0159 |
| DenseNet + FPN | 10.5M | 0.8891 ± 0.0168 | 0.8885 ± 0.0165 | 0.8696 ± 0.0162 | 0.8744 ± 0.0176 |
| **Proposed Architecture** | **10.5M** | **0.9040 ± 0.0195** | **0.9122 ± 0.0198** | **0.9130 ± 0.0123** | **0.9130 ± 0.0128** |

---

## **Intended Use**

This repository is intended for academic research and educational purposes.

The proposed framework is a research implementation and is not intended to replace professional medical diagnosis or clinical decision-making.
