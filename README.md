# 🧠 MNIST Digit Recognition Web App (Streamlit)

This is a real-time web application built using **Streamlit** that allows users to **draw a handwritten digit** (0–9) using a mouse or touch input, and get an instant prediction using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**.

---

## ✨ Features
- 🎨 **Canvas Drawing**: Draw digits in a web canvas interactively
- 🧼 **Smart Preprocessing**: Automatically centers and resizes the digit to match MNIST format
- 🔍 **Digit Prediction**: Uses a pre-trained CNN model to classify digits
- 📷 *(Optional)* Image Upload support via `utils/preprocess.py`

---

## 📁 Folder Structure
```
mnist_digit_recognition_app/
├── app.py                  # Main Streamlit app
├── models/
│   └── cnn_model.h5        # Pre-trained CNN model
├── utils/
│   └── preprocess.py       # (Optional) utility to preprocess uploaded images
├── requirements.txt        # All dependencies
└── README.md               # Project info and usage
```

---

## 📦 Installation & Setup

### 🔧 Step 1: Clone Repository
```bash
git clone https://github.com/ghafoor545/mnist_digit_recognition_app.git
cd mnist-digit-recognition-app
```

### 📦 Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### 📥 Step 3: Add Model File
Place your trained CNN model (`cnn_model.h5`) into the `models/` directory.

---

## 🚀 Run the Web App
```bash
streamlit run app.py
```

This will open a browser window. Draw a digit and press **Predict**.

---

## 🧠 Model Training (Optional)
If you don’t have a model yet, you can train one using TensorFlow/Keras:
```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(...)
model.save("models/cnn_model.h5")
```

---

## 🌐 Optional: Deploy Online
You can host this app for free using:
- [Streamlit Cloud](https://streamlit.io/cloud)
- [Hugging Face Spaces (Gradio/Streamlit)](https://huggingface.co/spaces)

Just upload your folder to GitHub and connect it to a platform.

---

## 🙌 Acknowledgements
- MNIST Dataset — Yann LeCun
- Streamlit Drawable Canvas — @andfanilo
- TensorFlow/Keras for deep learning

---

## 📬 Contact
For questions or improvements, feel free to reach out!

---

Made with ❤️ using Python, Streamlit, and Deep Learning.
