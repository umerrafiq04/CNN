# Face Mask Detection using Transfer Learning (MobileNetV2)

## Project Overview

This project implements a **Face Mask Detection system** using deep learning with **TensorFlow** and **Keras**, leveraging **transfer learning** with the **MobileNetV2** architecture. The system classifies whether a person is wearing a face mask or not, providing a foundation for real-time safety monitoring in public spaces.

The model is trained on a publicly available dataset from Kaggle containing images categorized as **with_mask** and **without_mask**.

---

## Key Features

* **Automated Dataset Handling**: Downloads and manages Kaggle datasets seamlessly.
* **Transfer Learning**: Utilizes MobileNetV2 pretrained on ImageNet for efficient feature extraction.
* **Custom Classifier**: Adds global average pooling, dropout regularization, and dense layers to classify images into two classes.
* **Optimized Performance**: Uses TensorFlow’s dataset caching and prefetching for faster training.
* **Scalable and Modular**: Designed for easy integration into real-time applications.

---

## Dataset

* **Source**: [Face Mask Dataset by Omkar Gurav](https://www.kaggle.com/omkargurav/face-mask-dataset)
* **Classes**:

  * `with_mask`
  * `without_mask`
* **Size**: 7,553 images (6,043 for training, 1,510 for validation)
* **Structure**:

  ```
  face-mask-dataset/
  ├── with_mask/
  └── without_mask/
  ```

---

## Technology Stack

* **Python 3.12**
* **TensorFlow 2.x / Keras**
* **MobileNetV2 (Pretrained on ImageNet)**
* **Kaggle API**
* **Google Colab** (for cloud-based execution)

---

## Installation

```bash
# Install Kaggle API
!pip install kaggle

# Set up Kaggle API credentials
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

---

## Usage

### 1. Dataset Download

```python
import kagglehub

# Download face mask dataset
path = kagglehub.dataset_download("omkargurav/face-mask-dataset")
print("Dataset path:", path)
```

### 2. Load Dataset

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

### 3. Model Architecture

* **Base Model**: MobileNetV2 (frozen weights)
* **Custom Layers**:

  * Global Average Pooling
  * Dropout (0.3)
  * Dense layer with 2 output classes and softmax activation

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

## Training

* **Train-validation split**: 80%-20%
* **Batch Size**: 32
* **Image Size**: 224x224
* **Data Prefetching**: Enabled for optimized performance

---

## Results

* The model is designed to achieve high accuracy in detecting whether a person is wearing a mask.
* Suitable for integration in **real-time surveillance systems**, public safety applications, and healthcare monitoring.

---

## Future Improvements

* Fine-tuning the base model for better accuracy.
* Implementing **real-time detection** using OpenCV.
* Adding support for **multiple face masks and occlusions**.
* Deploying as a web or mobile application with live camera input.

---

## License

This project is **open-source** under the MIT License.

---

### 🔗 Google Colab Notebook
You can run this project directly in Colab using the link below:

[Open in Google Colab](https://colab.research.google.com/drive/1Rk3d4QywT_1g9SXAxi8ZZedurCzi4MGG?usp=sharing)

## Contact

* **Developer**: Umer Rafiq
* **Email**: [rafiqyatooumer@gmail.com](mailto:raf@gmail.com)
  


