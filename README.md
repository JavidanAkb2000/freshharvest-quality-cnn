# 🍎 FreshHarvest AI Inspector

> **"Ensuring Quality, One Fruit at a Time."**

Welcome to **FreshHarvest AI Inspector**, an intelligent quality control system designed to automatically detect whether fruits are fresh or spoiled using state-of-the-art Computer Vision.

![FreshHarvest UI Demo](resources/ui_demo_preview.png)

---

## 🌟 Overview

FreshHarvest Logistics faces a challenge: manual inspection of fruits is slow, inconsistent, and prone to error. 

Our solution is a **Deep Learning-powered Inspector** that:
- 🚀 **Instant Analysis**: Classifies fruit in < 2 seconds.
- 🎯 **High Accuracy**: Achieves **98%+ accuracy** using Transfer Learning (EfficientNet-B0).
- 🤝 **User-Friendly**: Simple drag-and-drop interface for non-technical staff.

---

## 🧠 How It Works

The system is built on a robust **Convolutional Neural Network (CNN)** architecture. We leverage **Transfer Learning** with a pre-trained **EfficientNet-B0** model, fine-tuned on our custom dataset of fresh and spoiled fruits.

### Key Features:
- **Binary Classification**: Detects `Fresh` vs. `Spoiled`.
- **Confidence Scoring**: Provides a probability score for every prediction.
- **Visual Feedback**: Color-coded results (Green for Fresh, Red for Spoiled).

---

## 🍎 Supported Fruits

Our model is trained to recognize a variety of common produce:

| Fruit | Icon | Fruit | Icon |
|:------|:----:|:------|:----:|
| **Banana** | 🍌 | **Strawberry** | 🍓 |
| **Lemon** | 🍋 | **Tomato** | 🍅 |
| **Orange** | 🍊 | **Tamarillo** | 🫐 |
| **Mango** | 🥭 | **Lulo** | 🟢 |

---

## 💻 Installation & Usage

Follow these simple steps to get the Inspector running on your local machine.

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
Launch the Streamlit interface with a single command:
```bash
streamlit run src/app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`.

---

## 📂 Project Structure
```
FreshHarvest/
├── 📂 data/                 # Dataset of fresh/spoiled images
├── 📂 notebooks/            # Jupyter notebooks for training & experiments
│   ├── preparation.ipynb    # Data loading, preprocessing & training
│   └── experiments.ipynb    # Model testing & validation
├── 📂 src/                  # Source code for the application
│   ├── app.py               # Main Streamlit application
│   └── helper.py            # Helper functions (model loading, inference)
├── 📂 resources/            # Images and assets for documentation
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 📊 Model Performance

We evaluated our model on a held-out test set to ensure reliability.

| Metric | Score |
|:-------|------:|
| **Training Accuracy** | 99.1% |
| **Validation Accuracy** | 98.4% |
| **Test Accuracy** | 98%+ |
| **Inference Time** | ~150ms per image (CPU) |

*Note: Performance may vary slightly depending on hardware.*

---

## 🚀 Tech Stack

- **Deep Learning**: PyTorch, EfficientNet-B0
- **Frontend**: Streamlit
- **Computer Vision**: torchvision, PIL
- **Data Processing**: NumPy, Pandas

---

## 🤝 Contributing

We welcome feedback! If you have ideas for improvements or new features (like adding more fruit types), please feel free to open an issue or submit a pull request.

---

## 📝 License

This project is licensed under the MIT License.

---

<div align="center">
  <p>Made with ❤️ by the FreshHarvest AI Team</p>
  <p><i>Powered by PyTorch & Streamlit</i></p>
</div>