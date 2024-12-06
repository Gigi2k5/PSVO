import gradio as gr
import numpy as np
import tensorflow as tf
import cv2
import joblib
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# Charger le modèle de segmentation
segmentation_model = tf.keras.models.load_model('unet_optimized.keras', 
        custom_objects={"dice_coefficient": lambda y_true, y_pred: y_pred})

# Charger le modèle de classification
classification_model = joblib.load('knn.pkl')

# Classes pour le diagnostic
categories = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy']

def segment_image(image):
    # Redimensionner et normaliser l'image
    resized_image = cv2.resize(image, (256, 256)) / 255.0
    input_image = np.expand_dims(resized_image, axis=0)

    # Prédire le masque
    mask = segmentation_model.predict(input_image)[0]

    # Debugging : Visualiser les statistiques du masque
    print("Raw mask - Min:", np.min(mask), "Max:", np.max(mask), "Mean:", np.mean(mask))

    # Si nécessaire, normaliser le masque
    if np.max(mask) > 1.0:  # Si les valeurs sont hors de l'échelle attendue
        mask = mask / np.max(mask)

    # Seuillage pour obtenir une image binaire
    mask = (mask.squeeze() > 0.1).astype(np.uint8)

    # Debugging : Sauvegarder le masque binaire
    cv2.imwrite("binary_mask.png", mask * 255)

    # Redimensionner le masque à la taille originale
    original_size = (image.shape[1], image.shape[0])
    mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

    return mask_resized



# Fonction de classification
def classify_image(image):
    # Extraire les caractéristiques pour la classification
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    # Prédire la classe
    prediction = classification_model.predict([hist])
    return prediction[0]

# Fonction principale pour Gradio
def process_image(image):
    # Convertir l'image de PIL à NumPy
    image = np.array(image)

    # Segmentation
    mask = segment_image(image)

    # Classification
    diagnosis = classify_image(image)

    # Convertir le masque en image couleur pour l'affichage
    mask_colored = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

    return mask_colored, diagnosis

# Interface Gradio
interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(label="Chargez une image de feuille", type="pil"),
    outputs=[
        gr.Image(label="Masque de segmentation"),
        gr.Label(label="Diagnostic")
    ],
    title="SafeLeaf",
    description=(
        "Cette application est une application de détection des maladies des feuilles de pommiers, elle utilise deux modèles : "
        "1. Un modèle de segmentation pour détecter la zone de la feuille malade. "
        "2. Un modèle de classification pour diagnostiquer la maladie de la feuille. "
        "Chargez une image pour commencer."
    ),
)

# Lancer l'application
interface.launch()
