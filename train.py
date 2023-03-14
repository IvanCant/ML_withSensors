import tensorflow as tf
import pathlib
import numpy as np
import matplotlib.pyplot as plt


# go through all files in desired folder ###############################################################################
folders = ['./Traffic Signs/10 speed limit sign/', './Traffic Signs/15 speed limit sign/', './Traffic Signs/25 speed limit sign/', './Traffic Signs/30 speed limit sign/', './Traffic Signs/35 speed limit sign/', './Traffic Signs/45 speed limit sign/','./Traffic Signs/Slow sign/', './Traffic Signs/Stop sign/', './Traffic Signs/Wrong Way/', './Traffic Signs/Yield sign/', './Traffic Signs/Car/', './Traffic Signs/Pedestrians/', './Traffic Signs/Truck/']

img_height, img_width = [256, 256]
batch_size = 64
# POSSIBILY LARGER BATCH ###############################################################################################

# CHANGE ###############################################################################################################
data_dir = './Traffic Signs/'
data_dir = pathlib.Path(data_dir)

train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset="training", seed=123,
                                                       image_size=(img_height, img_width), batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset="validation", seed=123,
                                                     image_size=(img_height, img_width), batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

normalization_layer = tf.keras.layers.Rescaling(1. / 255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# CHANGE ###############################################################################################################
num_classes = 13

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'), #(filters, kernel_size,
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(.3),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(.3),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(.3),
    tf.keras.layers.Conv2D(256, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(.3),
    tf.keras.layers.Conv2D(256, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(.3),
    tf.keras.layers.Conv2D(256, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax'),
    #tf.keras.layers.Dense(num_classes)
])

#loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, validation_data=val_ds,
                    epochs=100)  # EPOCHS #############################################

y_vloss = history.history['val_loss']
y_loss = history.history['loss']
y_acc = history.history['accuracy']
y_vacc = history.history['val_accuracy']

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(np.arange(len(y_vloss)), y_vloss, marker='.', c='red')
ax1.plot(np.arange(len(y_loss)), y_loss, marker='.', c='blue')
ax1.grid()
plt.setp(ax1, xlabel='epoch', ylabel='loss')

ax2.plot(np.arange(len(y_vacc)), y_vacc, marker='.', c='red')
ax2.plot(np.arange(len(y_acc)), y_acc, marker='.', c='blue')
ax2.grid()
plt.setp(ax2, xlabel='epoch', ylabel='accuracy')

model.save('final20val.model')
plt.show()
