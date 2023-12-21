import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# Load and prepare dataset
def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label


# Build dataset
batch_size = 32
dataset_name = 'cifar10'
(ds_train, ds_test), ds_info = tfds.load(dataset_name, split=['train', 'test'], with_info=True, as_supervised=True)
ds_train = ds_train.map(preprocess_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Building the model
base_model = EfficientNetV2B0(weights='imagenet', include_top=False)
num_classes = ds_info.features['label'].num_classes
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)

# Callbacks
LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
TB = tf.keras.callbacks.TensorBoard('./files/cv_logs')
MC = tf.keras.callbacks.ModelCheckpoint(
    './files/cv_model/cv-{epoch:02d}-{val_loss:.3f}.h5',
    monitor='val_loss',
    save_best_only=False,
    verbose=1
)

# Initial training phase
base_model.trainable = False  # Freeze the backbone
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(ds_train, epochs=3, validation_data=ds_test, callbacks=[LR, TB, MC])

# Fine-tuning phase
model.trainable = True  # Make the whole model trainable
model.compile(optimizer=Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(ds_train, epochs=3, validation_data=ds_test, callbacks=[LR, TB, MC])


# Evaluate the model
loss, accuracy = model.evaluate(ds_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")
