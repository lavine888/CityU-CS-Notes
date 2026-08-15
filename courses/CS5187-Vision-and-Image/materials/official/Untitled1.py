
# coding: utf-8

# In[ ]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from PIL import Image
import cv2
import glob
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils.class_weight import compute_class_weight


# -------------------------- 1. Global Configurations & Utility Functions --------------------------
# Device configuration (GPU/CPU)
DEVICE_CONFIG = {
    "is_use_gpu": 1,
    "gpu_id": "1",
    "max_gpu_mem": 1.0,
    "input_size": 224  # Standard input size for pre-trained models like VGG16
}

# Path configuration (centralized management for easy modification)
PATH_CONFIG = {
    "raw_images": "./Images/",
    "raw_json": "./Images_json/",
    "queries_raw": "./Queries/",
    "cropped_queries": "./cropped_queries/",
    "cropped_images_yolo": "./cropped_images_yolo/",
    "cropped_resized_queries": "./cropped_resized_trans_queries/",
    "cropped_resized_images_yolo": "./cropped_resized_images_yolo/",
    "save_dir": "./save/",
    "result_dir": "./result/",
    "similarity_matrix": "./save/similarity_matrix.npy"  # Fixed original typo ("similaroty" → "similarity")
}

# Create all required directories (prevent "path not found" errors)
for dir_path in PATH_CONFIG.values():
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def sort_key_func(file_path):
    """Sort files by the numeric prefix in their filenames (e.g., "123_img.jpg" sorted by 123)"""
    return int(os.path.basename(file_path).split('_')[0])


def init_tensorflow_session():
    """Initialize TensorFlow session with GPU/CPU configuration"""
    if DEVICE_CONFIG["is_use_gpu"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_CONFIG["gpu_id"]
        device = '/device:GPU:0'
    else:
        device = "/cpu:0"

    # Session configuration (GPU memory management)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = DEVICE_CONFIG["max_gpu_mem"]
    config.gpu_options.allow_growth = True  # Dynamically allocate GPU memory
    config.allow_soft_placement = True      # Auto-switch to available devices
    config.log_device_placement = False     # Disable device logging to reduce clutter

    # Create session and bind to Keras backend
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    return sess, device


# -------------------------- 2. Image Preprocessing: Cropping & Resizing --------------------------
def crop_image_by_json(raw_img_dir, json_dir, output_crop_dir):
    """Crop images based on bounding boxes in JSON annotations and save to target directory"""
    img_paths = glob.glob(os.path.join(raw_img_dir, "*.jpg"))
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        img_no = img_name.split('.')[0]
        json_path = os.path.join(json_dir, f"{img_no}.json")

        # Read image (handle read failures)
        try:
            im = Image.open(img_path)
        except Exception as e:
            print(f"Failed to read image: {img_path}, Error: {e}")
            continue

        # Read JSON annotations and crop
        has_obj = False
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                try:
                    bbox_list = json.load(f)
                except json.JSONDecodeError:
                    print(f"Invalid JSON format: {json_path}")
                    bbox_list = []

            # Crop each bounding box
            for cnt, bbox in enumerate(bbox_list):
                topleft_x = bbox['topleft']['x']
                topleft_y = bbox['topleft']['y']
                bottomright_x = bbox['bottomright']['x']
                bottomright_y = bbox['bottomright']['y']

                # Save cropped region
                region = im.crop((topleft_x, topleft_y, bottomright_x, bottomright_y))
                region.save(os.path.join(output_crop_dir, f"{img_no}_{cnt}.jpg"))
                has_obj = True

        # Save original image if no bounding boxes exist
        if not has_obj:
            im.save(os.path.join(output_crop_dir, f"{img_no}_0.jpg"))
    print(f"Image cropping completed. Saved to: {output_crop_dir}")


def resize_cropped_images(input_crop_dir, output_resize_dir, target_size):
    """Resize all cropped images to a uniform target size"""
    img_paths = glob.glob(os.path.join(input_crop_dir, "*.jpg"))
    cnt = 0
    total = len(img_paths)

    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        # Read image (skip corrupted files)
        im = cv2.imread(img_path)
        if im is None:
            print(f"Skipping corrupted image: {img_path}")
            continue

        # Resize with bicubic interpolation (preserves details)
        im_resized = cv2.resize(im, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        # Save resized image
        cv2.imwrite(os.path.join(output_resize_dir, img_name), im_resized)

        # Print progress
        cnt += 1
        if cnt % 100 == 0 or cnt == total:
            print(f"Resize Progress: {cnt}/{total} ({cnt/total*100:.1f}%)")
    print(f"Image resizing completed. Saved to: {output_resize_dir}")


# -------------------------- 3. Transfer Learning: Train Image Classification Model --------------------------
def train_transfer_learning_model():
    """Train an image classification model using transfer learning with VGG16"""
    # Initialize TensorFlow session
    sess, device = init_tensorflow_session()

    # 1. Load pre-trained VGG16 (exclude top fully connected layers)
    base_model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(DEVICE_CONFIG["input_size"], DEVICE_CONFIG["input_size"], 3)
    )

    # 2. Data augmentation (for training set) and data loading
    train_datagen = ImageDataGenerator(
        rescale=1./255,          # Normalize pixel values to [0, 1]
        rotation_range=180,       # Random rotation (0-180 degrees)
        width_shift_range=0.1,    # Random horizontal shift (±10%)
        height_shift_range=0.1,   # Random vertical shift (±10%)
        shear_range=0.1,          # Random shear transformation (±10%)
        zoom_range=[0.9, 1.5],    # Random zoom (90%-150%)
        horizontal_flip=True,     # Random horizontal flip
        vertical_flip=True,       # Random vertical flip
        fill_mode='nearest'       # Fill new pixels with nearest neighbor
    )

    test_datagen = ImageDataGenerator(rescale=1./255)  # Only normalize test set

    # Load training data (assumes dataset is organized by class subdirectories)
    train_loader = train_datagen.flow_from_directory(
        directory=PATH_CONFIG["cropped_resized_queries"],
        target_size=(DEVICE_CONFIG["input_size"], DEVICE_CONFIG["input_size"]),
        batch_size=32,
        shuffle=True,
        save_to_dir=PATH_CONFIG["cropped_resized_queries"]  # Save augmented images
    )

    # Load test data (replace with independent test set in production)
    test_loader = test_datagen.flow_from_directory(
        directory=PATH_CONFIG["cropped_resized_queries"],
        target_size=(DEVICE_CONFIG["input_size"], DEVICE_CONFIG["input_size"]),
        batch_size=32,
        shuffle=False
    )

    # 3. Build transfer learning model
    model = Sequential([
        base_model,               # Pre-trained VGG16 convolutional layers
        Flatten(),                # Flatten convolutional features to 1D vector
        Dense(1024, activation='relu'),  # Custom fully connected layer
        Dropout(0.5),             # Dropout to prevent overfitting
        Dense(train_loader.num_classes, activation='softmax')  # Output layer (class count)
    ])

    # 4. Compile model (use small learning rate for fine-tuning)
    model.compile(
        optimizer=Adam(lr=1e-5),
        loss='categorical_crossentropy',  # Loss for multi-class classification
        metrics=['categorical_accuracy']  # Evaluation metric: classification accuracy
    )

    # 5. Training callback: Save the best model (highest training accuracy)
    checkpoint = ModelCheckpoint(
        os.path.join(PATH_CONFIG["save_dir"], "best_model.h5"),
        monitor='categorical_accuracy',
        verbose=1,
        save_best_only=True,
        mode='auto'
    )

    # 6. Compute class weights (address class imbalance)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_loader.classes),
        y=train_loader.classes
    )

    # 7. Start training
    history = model.fit_generator(
        generator=train_loader,
        epochs=40,
        steps_per_epoch=10,
        class_weight=class_weights,
        validation_data=test_loader,
        validation_steps=test_loader.n // 32,
        callbacks=[checkpoint]
    )

    # 8. Save training history plot and final model
    # Plot training curves
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(acc, 'b-', label='Train Accuracy')
    plt.plot(val_acc, 'r--', label='Test Accuracy')
    plt.plot(loss, 'bo', label='Train Loss')
    plt.plot(val_loss, 'ro', label='Test Loss')
    plt.legend()
    plt.title('Training History')
    plt.savefig(os.path.join(PATH_CONFIG["save_dir"], "train_history.png"))
    plt.close()

    # Save final model
    model.save(os.path.join(PATH_CONFIG["save_dir"], "final_model.h5"))
    print("Model training completed. Saved to: ", PATH_CONFIG["save_dir"])
    return model


# -------------------------- 4. Feature Extraction & Similarity Calculation --------------------------
def extract_features(model_path, input_img_dir, target_size):
    """Load a trained model and extract image features (from 'flatten' layer)"""
    # Load trained model and extract 'flatten' layer output as features
    trained_model = load_model(model_path)
    feature_model = Model(
        inputs=trained_model.input,
        outputs=trained_model.get_layer('flatten').output
    )

    # Read images and extract features
    img_paths = glob.glob(os.path.join(input_img_dir, "*.jpg"))
    img_paths.sort(key=sort_key_func)  # Sort by image ID for consistency
    features = []

    for img_path in img_paths:
        im = cv2.imread(img_path)
        if im is None:
            print(f"Skipping corrupted image: {img_path}")
            continue
        # Preprocess: Normalize + add batch dimension
        im = im / 255.0
        im = np.expand_dims(im, axis=0)
        # Extract features and flatten to 1D vector
        feat = feature_model.predict(im)
        features.append(feat.flatten())

    return np.array(features), img_paths


def compute_cosine_similarity(feat1, feat2):
    """Compute cosine similarity between two feature matrices (after L2 normalization)"""
    # L2 normalization (normalize feature vectors to unit sphere)
    feat1_norm = tf.nn.l2_normalize(feat1, axis=1)
    feat2_norm = tf.nn.l2_normalize(feat2, axis=1)
    # Calculate similarity (dot product of normalized features)
    similarity = tf.matmul(feat1_norm, feat2_norm, transpose_b=True)
    # Run computation in current session
    sess = tf.keras.backend.get_session()
    return sess.run(similarity)


# -------------------------- 5. Similarity Matrix Aggregation & Retrieval Ranking --------------------------
def aggregate_similarity(sim_matrix, img_dir):
    """Aggregate similarity scores for multiple crops of the same image (take max value)"""
    # Get list of image IDs from filenames
    img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
    img_paths.sort(key=sort_key_func)
    img_ids = [int(os.path.basename(p).split('_')[0]) for p in img_paths]
    img_ids = np.array(img_ids)

    # Get unique image IDs, their first occurrence indices, and crop counts
    unique_ids, first_idx, counts = np.unique(img_ids, return_index=True, return_counts=True)
    n_unique = len(unique_ids)

    # Initialize aggregated similarity matrix
    if sim_matrix.ndim == 2:
        n_row = sim_matrix.shape[0]
        n_col = n_unique
        agg_sim = np.zeros((n_row, n_col))
    else:
        n_row = n_unique
        n_col = sim_matrix.shape[1]
        agg_sim = np.zeros((n_row, n_col))

    # Aggregate scores (max value for multiple crops of the same image)
    for uid, idx, cnt in zip(unique_ids, first_idx, counts):
        if cnt == 1:
            if sim_matrix.ndim == 2:
                agg_sim[:, uid-1] = sim_matrix[:, idx]  # Assume IDs start from 1
            else:
                agg_sim[uid-1, :] = sim_matrix[idx, :]
        else:
            if sim_matrix.ndim == 2:
                dup_scores = sim_matrix[:, idx:idx+cnt]
                agg_sim[:, uid-1] = np.max(dup_scores, axis=1)
            else:
                dup_scores = sim_matrix[idx:idx+cnt, :]
                agg_sim[uid-1, :] = np.max(dup_scores, axis=0)
    return agg_sim, unique_ids


def generate_rank_list(similarity_matrix, save_path):
    """Generate a similarity ranking list and save to a text file"""
    # Sort by similarity in descending order (return indices)
    rank = np.argsort(similarity_matrix, axis=-1)[:, ::-1]

    # Save ranking to file
    with open(save_path, 'w') as f:
        for q_idx, row in enumerate(rank, 1):
            rank_str = " ".join([str(i) for i in row])
            f.write(f"Q{q_idx}: {rank_str}\n")
    print(f"Ranking list saved to: {save_path}")
    return rank


# -------------------------- 6. Retrieval Result Visualization --------------------------
def visualize_retrieval_result(rank, top_k=10, n_query=5):
    """Visualize Top-k retrieval results for the first n_query queries"""
    # Select Top-k results for the first n_query queries
    top_rank = rank[:n_query, :top_k].flatten()
    # Create plot (5 rows × 11 columns: 5 queries + 5×10 results)
    fig, axes = plt.subplots(5, 11, figsize=(22, 10))
    axes = axes.flatten()  # Flatten axes array for easy indexing
    cnt = 0

    # Plot queries and their results
    for idx in top_rank:
        # Plot query image (1st position in every 11 subplots)
        if cnt % 11 == 0:
            q_id = cnt // 11 + 1
            # Generate query image filename (e.g., "01.jpg" for query 1)
            q_img_name = f"0{q_id % 10}.jpg" if q_id <= 10 else f"{q_id}.jpg"
            q_img_path = os.path.join(PATH_CONFIG["queries_raw"], q_img_name)
            if os.path.exists(q_img_path):
                q_im = mpimg.imread(q_img_path)
                axes[cnt].imshow(q_im)
            axes[cnt].set_title(f"Query {q_id}")
            axes[cnt].axis('off')  # Hide axes
            cnt += 1

        # Plot retrieval result image (with red bounding boxes)
        img_id = idx + 1  # Convert index to image ID (IDs start from 1)
        #

