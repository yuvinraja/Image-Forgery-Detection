# Image Forgery Detection using CNN

This is a **Streamlit** web application for detecting image forgery using **Error Level Analysis (ELA)** and a **Convolutional Neural Network (CNN)**. The model is trained on CASIA2 dataset and is uploaded to Hugging Face and predicts whether an uploaded image is **real or fake**.

## 🚀 Features

- Upload an image for forgery detection.
- Convert the image to an **ELA (Error Level Analysis) format**.
- Load a **pre-trained CNN model** for forgery detection.
- Display the model's **prediction** along with a **confidence score**.

---

## 🛠 Installation & Running the App Locally

Follow these steps to set up and run the app on your local machine:

### 1️⃣ Clone the Repository

```sh
git clone https://github.com/your-username/image-forgery-detection.git
cd image-forgery-detection
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)

```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### 3️⃣ Install Dependencies

```sh
pip install -r requirements.txt
```

### 4️⃣ Run the App

```sh
streamlit run app.py
```

The app will launch in your default web browser.

---

## 🔍 How It Works

1. **Model Download**: The app downloads a pre-trained **CNN model** from Hugging Face.
2. **Image Preprocessing**: The uploaded image is converted into an **ELA format**.
3. **Prediction**: The processed image is passed to the model, which predicts whether the image is **real or fake**.
4. **Results Display**: The app shows the prediction and confidence score.

---

## 📷 Example

When an image is uploaded, the app displays:

- The **uploaded image** 📷
- The **ELA-transformed image** 🎨
- The **forgery prediction** and **confidence score** ✅❌

---

## 🔗 Model Details

- **Model Repository**: [Hugging Face Model](https://huggingface.co/yuvinraja/image-forgery-model)
- **Model File**: `model_casia_run1.h5`

---

## 🛠 Troubleshooting

### Common Issues & Fixes

- **Model Download Error (RepositoryNotFoundError)**:
  - Ensure the **Hugging Face repository** is public or that you have provided the correct authentication credentials.
  - Double-check `MODEL_REPO` and `MODEL_FILENAME` in `app.py`.
- **ModuleNotFoundError**: Run `pip install -r requirements.txt` again.

---

## 📌 Future Improvements

- Add **user authentication** for better security.
- Improve **model accuracy** using a more advanced CNN architecture.
- Deploy the app on **Streamlit Cloud** for easy access.

---

## 🤝 Contributing

Feel free to **fork** the repository and submit a **pull request** with your improvements!

