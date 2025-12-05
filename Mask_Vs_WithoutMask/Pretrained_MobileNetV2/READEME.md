# 😷 Face Mask Detection using Transfer Learning (MobileNetV2)

## 📌 Project Overview

This project implements a **Face Mask Detection system** using deep learning with **TensorFlow** and **Keras**, leveraging **transfer learning** with the **MobileNetV2** architecture.
The system classifies whether a person is **wearing a face mask** or **not**, making it suitable for real-time safety monitoring in public environments.

The model is trained on a publicly available Kaggle dataset containing two categories: **with_mask** and **without_mask**.

---

## ⭐ Key Features

* **Automated Dataset Handling** – Easily downloads and organizes the dataset from Kaggle.
* **Transfer Learning** – Uses MobileNetV2 pretrained on ImageNet for efficient feature extraction.
* **Custom Classification Head** – Includes global average pooling, dropout, and dense layers.
* **Optimized Training** – Utilizes caching and prefetching for faster data loading.
* **Modular & Scalable** – Can be integrated into real-time detection systems.

---

## 📂 Dataset

**Source:** *Face Mask Dataset by Omkar Gurav (Kaggle)*

**Classes:**

* `with_mask`
* `without_mask`

**Size:**

* Total images: **7,553**
* Training: **6,043**
* Validation: **1,510**

**Directory Structure:**

```
face-mask-dataset/
├── with_mask/
└── without_mask/
```

---

## 🛠️ Technology Stack

* Python **3.12**
* TensorFlow **2.x** / Keras
* MobileNetV2 (ImageNet pretrained)
* Kaggle API
* Google Colab (GPU-supported)

---

## 📥 Installation

```bash
# Install Kaggle API
!pip install kaggle

# Set up Kaggle credentials
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

---

## 🚀 Usage

### **1️⃣ Download the Dataset**

```python
import kagglehub

# Download face mask dataset
path = kagglehub.dataset_download("omkargurav/face-mask-dataset")
print("Dataset path:", path)
```

---

### **2️⃣ Load Dataset**

```python
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path + '/data',
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path + '/data',
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)
```

---

### **3️⃣ Model Architecture**

#### **Base Model:** MobileNetV2 (Frozen)

#### **Custom Layers:**

* Global Average Pooling
* Dropout (0.3)
* Dense layer (softmax activation)

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

base_model = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.summary()
```

---

## 🏋️ Training Details

* **Train/Validation Split:** 80% / 20%
* **Batch Size:** 32
* **Image Size:** 224×224
* **Performance Optimization:** dataset caching & prefetching enabled

---

## 📈 Results

* Achieves high accuracy in detecting mask vs. no mask.
* Suitable for integration into:

  * Real-time surveillance
  * Public safety applications
  * Healthcare monitoring systems

---

## 🔮 Future Improvements

* Fine-tune MobileNetV2 layers for higher accuracy
* Real-time video detection using OpenCV
* Support for occluded or partially covered faces
* Deployment on web/mobile platforms

---

## 📄 License

This project is open-source under the **MIT License**.

---

## 🔗 Google Colab Notebook

Run the complete project in Google Colab:

👉 *(https://colab.research.google.com/drive/1Rk3d4QywT_1g9SXAxi8ZZedurCzi4MGG?usp=sharing)*

---

## 📬 Contact

**Developer:** Umer Rafiq
**Email:** [rafiqyatooumer@gmail.com](mailto:rafiqyatooumer@gmail.com)


