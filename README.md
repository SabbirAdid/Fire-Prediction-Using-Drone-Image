
# Fire Detection Using Satellite Images

## Overview
This project uses deep learning techniques for fire detection in satellite images. The model utilizes **ResNet50** as the base model with transfer learning to classify images as either "fire" or "non-fire". The dataset used contains images from various sources and is organized into training, validation, and testing sets.

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- OpenCV (optional, for image visualization)
- Numpy

You can install the necessary dependencies using pip:
```bash
pip install tensorflow numpy opencv-python
```

## Dataset
The dataset is organized into three directories:
- `train`: Training images
- `val`: Validation images
- `test`: Testing images

Each of these directories contains subdirectories for each class (`fire` and `non-fire`).

## Steps Involved

### Step 1: Library Import
We start by importing the necessary libraries, such as TensorFlow, Keras, and image processing tools.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import os
```

### Step 2: Dataset Location
Specify the path to your dataset.

```python
dataset_path = "/content/fire_split_dataset"
```

### Step 3: Load Images Using ImageDataGenerator
We use Keras' `ImageDataGenerator` to load and preprocess the images. Data augmentation techniques like rotation and zooming are applied to the training set to prevent overfitting.

```python
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, zoom_range=0.1, horizontal_flip=True)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(os.path.join(dataset_path, "train"), target_size=(250, 250), batch_size=32, class_mode='binary')
val_generator = val_test_datagen.flow_from_directory(os.path.join(dataset_path, "val"), target_size=(250, 250), batch_size=32, class_mode='binary')
test_generator = val_test_datagen.flow_from_directory(os.path.join(dataset_path, "test"), target_size=(250, 250), batch_size=32, class_mode='binary', shuffle=False)
```

### Step 4: Model Creation Using ResNet50
We use **ResNet50** as the base model, excluding the top classification layers. A new custom classification head is added on top of the base model.

```python
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(250, 250, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
```

### Step 5: Freeze Base Model and Train New Layers
We freeze the layers of the base model and only train the new layers that we have added.

```python
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
```

### Step 6: Model Training
Train the model using the training dataset and validate using the validation dataset.

```python
model.fit(train_generator, epochs=10, validation_data=val_generator)
```

### Step 7: Model Evaluation and Prediction
After training, evaluate the model on the test set and make predictions.

```python
predictions = model.predict(test_generator)
```

## License
This project is open-source and available under the MIT License.

## Acknowledgments
- **TensorFlow** and **Keras** for providing deep learning tools.
- **ResNet50** for pre-trained model weights used in transfer learning.
- Dataset contributors for providing labeled satellite images.
