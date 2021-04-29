import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np


_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)

"""
#cats and directory structure as follows

/root/.keras/datasets
/root/.keras/datasets/cats_and_dogs_filtered
/root/.keras/datasets/cats_and_dogs_filtered/validation
/root/.keras/datasets/cats_and_dogs_filtered/validation/cats
/root/.keras/datasets/cats_and_dogs_filtered/validation/dogs
/root/.keras/datasets/cats_and_dogs_filtered/train
/root/.keras/datasets/cats_and_dogs_filtered/train/cats
/root/.keras/datasets/cats_and_dogs_filtered/train/dogs

"""
base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')


BATCH_SIZE = 100
IMG_SHAPE = 150

#Overfitting is one of the major problem in ML, the model memorise the train model and due lacks the generalization on the data
# if validation data accuracy is less compared to training data accuracy, then model can be considered as Overfitting
#Data Augmentation -- Images are rotated or zoomed or flip in horizontal or vertical direction in order to aviod overfit

#Dropout -- Turning off certain neurons duirng propagation, so that model doesn't overfit
#Normalization -- Bringing all images on to same scale (0,1)

train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)

image_augmentation = ImageDataGenerator(
                        rescale=1./255,
                        rotation_range = 40,
                        width_shift_range =0.2,
                        height_shift_range=0.2,
                        shear_range =0.2,
                        zoom_range = 0.2,
                        horizontal_flip=True,
                        fill_mode ='nearest'
)

# we do augmentation only for training data

train_img_gen = image_augmentation.flow_from_directory(batch_size = BATCH_SIZE,directory=train_dir,
                                                          shuffle=True,class_mode='binary',
                                                          target_size = (IMG_SHAPE,IMG_SHAPE))
valid_img_gen = validation_image_generator.flow_from_directory(batch_size = BATCH_SIZE,directory=validation_dir,
                                                          shuffle=True,class_mode='binary',
                                                          target_size = (IMG_SHAPE,IMG_SHAPE))
class_names = ['cat','dog']

model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32,(3,3),input_shape=(150,150,3),activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),

            tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),

            tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),

            tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Dropout(0.5), #0.5 means 50% neurons are dropped
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512,activation='relu'),
            tf.keras.layers.Dense(2)
])

model.compile(optimizer='adam',
             loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics = ['accuracy'],
)

model.summary()

total_trian_imgs = 2000
total_valid_imgs = 1000

trained_model =model.fit(
        train_img_gen,
        steps_per_epoch = int(np.ceil(total_trian_imgs / float(BATCH_SIZE))),
        epochs= 100,
        validation_data = valid_img_gen,
        validation_steps = int(np.ceil(total_valid_imgs/float(BATCH_SIZE)))
)

dog_image = "file:///C:/Users/meher/Downloads/dog.jpg"
dog_image_path = tf.keras.utils.get_file('dog', origin=dog_image)

cat_image = "file:///C:/Users/meher/Downloads/cat_1.jpg"
cat_image_path = tf.keras.utils.get_file('cat', origin=cat_image)

img = tf.keras.preprocessing.image.load_img(
    cat_image_path, target_size=(IMG_SHAPE, IMG_SHAPE)
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
print('predications', predictions)
