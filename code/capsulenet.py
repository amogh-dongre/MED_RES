import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import InputLayer,Conv2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Layer
from keras.layers import Reshape
from keras.layers import Lambda
from keras import initializers
import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def preprocessing(path):
    img = Image.open(path)
    img_resized = img.resize((224,224))
    return np.array(img_resized)/255

data_path = '../data/spectrograms'
class0 = 'class_0'
class1 = 'class_1'
non_seizure = []
seizure = []
n_seizure_names = [f for f in os.listdir(os.path.join(data_path,class0))]
seizure_names = [f for f in os.listdir(os.path.join(data_path,class1))]

for path in n_seizure_names:
    non_seizure.append(preprocessing(os.path.join(data_path,class0,path)))
for path in seizure_names:
    seizure.append(preprocessing(os.path.join(data_path,class1,path)))

non_seizure_labels = [0]*len(non_seizure)
seizure_labels = [1]*len(seizure)
data = non_seizure+seizure
data_labels = non_seizure_labels+seizure_labels

X, y = shuffle(np.array(data), np.eye(2)[data_labels], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=1/3, random_state=42)

def squash(inputs):
    squared_norm = K.sum(K.square(inputs), axis=-1, keepdims=True)
    return (squared_norm / (1 + squared_norm)) / (K.sqrt(squared_norm + K.epsilon())) * inputs

class PrimaryCapsuleLayer(Layer):
    def __init__(self, num_capsules, capsule_dim, **kwargs):
        super(PrimaryCapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        
    def build(self, input_shape):
        super(PrimaryCapsuleLayer, self).build(input_shape)
        
    def call(self, inputs):
        batch_size = K.shape(inputs)[0]
        reshaped = K.reshape(inputs, (batch_size, self.num_capsules, self.capsule_dim))
        return squash(reshaped)
    
    def compute_output_shape(self, input_shape):
        return (None, self.num_capsules, self.capsule_dim)

class DigitCapsuleLayer(Layer):
    def __init__(self, num_classes=2, capsule_dim=16, routings=3, **kwargs):
        super(DigitCapsuleLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.capsule_dim = capsule_dim
        self.routings = routings
        self.kernel_initializer = initializers.get('glorot_uniform')
    
    def build(self, input_shape):
        self.num_primary_capsules = int(input_shape[1])
        self.primary_capsule_dim = int(input_shape[2])
        
        self.W = self.add_weight(
            shape=(1, self.num_primary_capsules, self.num_classes, self.primary_capsule_dim, self.capsule_dim),
            initializer=self.kernel_initializer,
            name='weights'
        )
        self.built = True
    
    def call(self, inputs):
        batch_size = K.shape(inputs)[0]
        
        W_tiled = K.tile(self.W, [batch_size, 1, 1, 1, 1])
        
        inputs_expanded = K.expand_dims(K.expand_dims(inputs, 2), -1)
        inputs_tiled = K.tile(inputs_expanded, [1, 1, self.num_classes, 1, 1])
        
        # Matrix multiplication: (batch_size, num_primary_capsules, num_classes, capsule_dim)
        u_hat = K.sum(inputs_tiled * W_tiled, axis=3)
        
        b = tf.zeros(shape=[batch_size, self.num_primary_capsules, self.num_classes])
        
        for i in range(self.routings):
            c = K.softmax(b, axis=-1)  # (batch_size, num_primary_capsules, num_classes)
            
            c_expanded = K.expand_dims(c, -1)  # (batch_size, num_primary_capsules, num_classes, 1)
            s = K.sum(c_expanded * u_hat, axis=1)  # Sum over primary capsules
            
            # Squash to get output capsules
            v = squash(s)  # (batch_size, num_classes, capsule_dim)
            
            # Update routing coefficients (except for last iteration)
            if i < self.routings - 1:
                v_expanded = K.expand_dims(v, 1)
                agreement = K.sum(u_hat * v_expanded, axis=-1)
                b = b + agreement
        
        return v
    
    def compute_output_shape(self, input_shape):
        return (None, self.num_classes, self.capsule_dim)

def output_layer(inputs):
    return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())

def mask(outputs):
    if type(outputs) != list:
        norm_outputs = K.sqrt(K.sum(K.square(outputs), -1) + K.epsilon())
        y = K.one_hot(indices=K.argmax(norm_outputs, 1), num_classes=2)
        y = K.expand_dims(y, -1)  # (batch_size, num_classes, 1)
        return Flatten()(y * outputs)
    else:
        y = K.expand_dims(outputs[1], -1)  # (batch_size, num_classes, 1)
        masked_output = y * outputs[0]
        return Flatten()(masked_output)

def loss_fn(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))

input_tensor = Input(shape=(224, 224, 4))

conv1 = Conv2D(256, (9, 9), activation='relu', padding='valid')(input_tensor)
conv2 = Conv2D(256, (9, 9), strides=2, activation='relu', padding='valid')(conv1)

conv2_reshaped = Conv2D(256, (9, 9), strides=2, activation='relu', padding='valid')(conv2)

primary_capsules_dim = 8
num_primary_capsules = (48 * 48 * 256) // primary_capsules_dim

primary_caps = PrimaryCapsuleLayer(
    num_capsules=num_primary_capsules, 
    capsule_dim=primary_capsules_dim
)(Flatten()(conv2_reshaped))

# Digit capsules
digit_caps = DigitCapsuleLayer(num_classes=2, capsule_dim=16, routings=3)(primary_caps)

# Output layer
outputs = Lambda(output_layer)(digit_caps)

# Masking for decoder
inputs_label = Input(shape=(2,))
masked = Lambda(mask)([digit_caps, inputs_label])
masked_for_test = Lambda(mask)(digit_caps)

# Decoder
decoded_inputs = Input(shape=(2 * 16,))  # 2 classes * 16 dim
dense1 = Dense(512, activation='relu')(decoded_inputs)
dense2 = Dense(1024, activation='relu')(dense1)
decoded_outputs = Dense(224 * 224 * 4, activation='sigmoid')(dense2)
decoded_outputs = Reshape((224, 224, 4))(decoded_outputs)

decoder = Model(decoded_inputs, decoded_outputs)

# Final models
model = Model([input_tensor, inputs_label], [outputs, decoder(masked)])
test_model = Model(input_tensor, [outputs, decoder(masked_for_test)])

print("Model created successfully!")
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=[loss_fn, 'mse'],
    loss_weights=[1., 0.0005],
    metrics=['accuracy' , None]
)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'weights/CapsuleNet.weights.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True
)

# Train the model
history = model.fit(
    [X_train, y_train],  
    [y_train, X_train],  
    validation_data=([X_val, y_val], [y_val, X_val]),
    epochs=30,
    batch_size=20,
    callbacks=[early_stopping, checkpoint]
)
