# Deep Learning-Based IDS for Resource-Constrained Networks

🛡️ **An optimized Intrusion Detection System designed for high-performance security on edge devices and IoT nodes.**

## Project Overview

This repository contains the source code for a Deep Learning-based Intrusion Detection System (IDS) developed specifically for networks with **resource constraints**. The core challenge addressed is balancing detection accuracy with the computational and memory limitations of edge hardware.

The project implements two distinct model architectures—a **Full Deep Neural Network (DNN)** and a custom **Lightweight DNN**—to provide a side-by-side performance benchmark on the NSL-KDD benchmark dataset.

---

## 🚀 Key Results & Impact

The simulation demonstrates that significant reductions in complexity can be achieved with negligible security loss.

| Metric | Full DNN (Baseline) | **Lightweight DNN (Optimized)** | **Improvement** |
| :--- | :--- | :--- | :--- |
| **Detection Accuracy** | 80.90% | **80.05%** | *-0.85% (Maintained)* |
| **Network Parameters** | 13,057 | **3,457** | **~74% Reduction** |
| **Training Time** | 185.78s | **44.48s** | **~76% Faster** |

### **Conclusion for Edge Deployment**
The Lightweight model achieved a **~74% reduction in memory complexity** and was **~76% faster to train**, making it a highly viable candidate for deployment on low-power hardware (e.g., IoT gateways, sensor nodes) while maintaining an **80.05% detection accuracy**.

---

## 🛠️ Technical Stack & Implementation

- **Language:** Python (v3.9+)
- **Data Handling & Analysis:** Pandas, NumPy
- **Model Development:** Scikit-Learn (specifically `MLPClassifier` for DNN implementation)
- **Data Visualization:** Matplotlib
- **Interactive Dashboard:** Streamlit

### **Feature Engineering Workflow**
The project focuses heavily on **efficient data preprocessing** to reduce data dimensionality before model input, which is critical for constrained systems:
1.  **Label Encoding** of categorical network data.
2.  **MinMax Scaling** for consistent data ranges.
3.  **Univariate Feature Selection (`SelectKBest`)** to identify and keep only the top `N` most relevant network traffic features, reducing the input data footprint by over 60%.

---

## 📂 Repository Structure

```text
├── app.py                # Streamlit source code for the interactive dashboard UI.
├── main.py               # Backend logic: Preprocessing, model training, and simulation.
├── requirements.txt      # List of project dependencies.
└── .gitignore            # Rules for excluding unnecessary files from repository.


Running Locally
1. Prerequisites
Python 3.9+ installed.

The NSL-KDD dataset downloaded and placed in a folder named NSL-KDD at the root directory.

2. Setup & Execution


i. Clone the repository:

	git clone [https://github.com/merezki-11/Deep-Learning-Based-IDS-for-Resource-Constrained-Networks.git]
	(https://github.com/merezki-11/Deep-Learning-Based-IDS-for Resource-Constrained-Networks.git)
	cd Deep-Learning-Based-IDS-for-Resource-Constrained-Networks

ii. Install dependencies:

	pip install -r requirements.txt

iii. Run the Simulation Backend:
(This trains the models and prints a statistical summary to the terminal)

	python main.py

iv. Launch the Interactive Dashboard:
	streamlit run app.py


