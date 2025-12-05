# 🟦 Face Mask Detection using Custom CNN

## 📌 Project Overview

This project implements a **Face Mask Detection system** using a **Convolutional Neural Network (CNN)** trained from scratch.
The model learns to classify images into two categories:

* **With Mask**
* **Without Mask**

The entire pipeline — data preprocessing, augmentation, model building, training, and evaluation — is implemented using **TensorFlow** and **Keras**.

---

## ⭐ Key Features

✔ **Custom CNN model designed and trained from scratch**
✔ **Data augmentation for better generalization**
✔ **Training with callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)**
✔ **Handles dataset automatically using `flow_from_directory`**
✔ **High validation accuracy**
✔ **User image upload support**
✔ **Real-time mask detection using webcam (Colab-supported)**

---

## 📂 Dataset

* **Source**: Face Mask Dataset (Kaggle)
* **Classes**:

  * `with_mask`
  * `without_mask`
* **Directory Structure**:

```
dataset/
├── with_mask/
└── without_mask/
```

---

## 🛠️ Technologies Used

* Python 3.x
* TensorFlow / Keras
* OpenCV (optional)
* NumPy
* Google Colab

---

## 📥 Setup & Installation

```bash
!pip install tensorflow opencv-python
```

If using Kaggle:

```bash
!pip install kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

---

## 📊 Data Loading & Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    shear_range=0.1,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    data_path,
    target_size=(128,128),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

val_data = train_gen.flow_from_directory(
    data_path,
    target_size=(128,128),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)
```

---

## 🧠 Model Architecture (Scratch CNN)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(128,128,3)))
model.add(MaxPooling2D())

model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D())

model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))

model.summary()
```

---

## ⏳ Training with Callbacks

```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.3),
    ModelCheckpoint("best_mask_model.h5", save_best_only=True)
]

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=callbacks
)
```

---

## 📈 Results

Example output:

```
48/48 ━━━━━━━━━━━━━━━━━━━━ 7s 146ms/step
accuracy: 0.9682 - loss: 0.0660
Validation Accuracy: 0.9748
Validation Loss: 0.0581
```

This reflects strong training performance for a scratch CNN model.

---

## 🖼️ View Predictions on First 10 Validation Images

```python
import matplotlib.pyplot as plt
import numpy as np

x, y = next(val_data)

for i in range(10):
    img = x[i]
    pred = model.predict(img.reshape(1,128,128,3))[0][0]
    label = "With Mask" if pred > 0.5 else "Without Mask"

    plt.imshow(img)
    plt.title(f"Prediction: {label}")
    plt.axis("off")
    plt.show()
```

---

## 📤 Predict Uploaded Images

```python
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img("/content/test.jpg", target_size=(128,128))
img = image.img_to_array(img)/255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
print("Prediction:", "With Mask" if pred > 0.5 else "Without Mask")
```
## 🚀 Future Enhancements

* Add face detection before classification
* Improve regularization
* Experiment with deeper architectures
* Deploy the model as a web or mobile app

---

## 📜 License

This project is licensed under the **MIT License**.

---
## 🔗 Google Colab Notebook

Run the complete project in Google Colab:

👉 *(https://colab.research.google.com/drive/1EVJWpIr18xGZL-iDjLFgd1kQQ2Ra8PS2?usp=sharing)*

---

## ✉️ Contact

**Developer:** Umer Rafiq
**Email:** [rafiqyatooumer@gmail.com](mailto:rafiqyatooumer@gmail.com)
