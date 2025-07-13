import streamlit as st

st.title("📊 Model Comparison and Info")

st.markdown("""
### 🧪 Models Trained
- ✅ CNN (Convolutional Neural Network)
- ✅ SVM (Support Vector Machine)
- ✅ k-NN (K-Nearest Neighbors)
- ✅ Random Forest Classifier

### 🎯 Evaluation Metrics Used
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

### 🔍 Dataset
- **MNIST**: 60,000 training images + 10,000 test images
- Grayscale handwritten digits (0–9), size 28x28 pixels
""")
