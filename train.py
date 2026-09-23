import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models

# ==========================================
# STEP 1: Load and Preprocess the Images
# ==========================================
# Point to where your extracted data is located
TRAIN_DIR = "plant_diseases_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"
VALID_DIR = "plant_diseases_dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid"

IMAGE_SIZE = (224, 224) # MobileNetV2 expects 224x224 images
BATCH_SIZE = 32

print("Loading training data...")
train_dataset = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

print("Loading validation data...")
validation_dataset = image_dataset_from_directory(
    VALID_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Optimize loading speeds
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)


# ==========================================
# STEP 2: Configure Transfer Learning Model
# ==========================================
# 1. Load MobileNetV2 pre-trained on millions of everyday images (ImageNet)
# include_top=False removes the default final 1,000-class classifier layer
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# 2. FREEZE THE BASE LAYER: Lock the weights so we don't destroy pre-trained knowledge
base_model.trainable = False

# 3. Create your custom model architecture
model = models.Sequential([
    # Inputs go into MobileNetV2 preprocessing
    layers.Input(shape=(224, 224, 3)),
    layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input),
    
    # Send data through the frozen pre-trained model
    base_model,
    
    # Flatten spatial features into vectors
    layers.GlobalAveragePooling2D(),
    
    # Add a Dropout layer to prevent overfitting
    layers.Dropout(0.2),
    
    # Final layer: Outputs predictions matching your exact dataset classes
    layers.Dense(len(train_dataset.class_names), activation='softmax')
])


# ==========================================
# STEP 3: Compile and Train
# ==========================================
model.compile(
    optimizer='adam',
    loss='categorical_with_cross_entropy', # Good for multi-class classification
    metrics=['accuracy']
)

model.summary() # Displays the setup layout of your neural network

print("\nStarting training... (Try 3 epochs to start)")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=3 
)

# Save your trained model locally
model.save("plant_disease_mobilenet.h5")
print("Model trained and saved successfully!")
