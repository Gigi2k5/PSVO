import gradio as gr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
import io
from PIL import Image


def segment_image(image):
    
    img = np.array(image)
    
    twoDim = img.reshape((-1, 3))
    
    twoDim = np.float32(twoDim)
    
    mean_shift = MeanShift(bandwidth=30, bin_seeding=True)  
    mean_shift.fit(twoDim)
    
    labels = mean_shift.labels_
  
    segmented_image = mean_shift.cluster_centers_[labels].astype(np.uint8)
    
    segmented_image = segmented_image.reshape(img.shape)
    
    segmented_image_pil = Image.fromarray(segmented_image)
    return segmented_image_pil

def analyze_health(image):
    
    return "Plante en bonne santé, pas de signes de stress ou de maladie."

# Interface Gradio
def plant_diagnosis(image):
    
    segmented = segment_image(image)
    
    diagnosis = analyze_health(segmented)
    
    return segmented, diagnosis

# Créer l'interface Gradio
iface = gr.Interface(
    fn=plant_diagnosis, 
    inputs=gr.Image(type="pil", label="Téléchargez une image de votre plante"),  
    outputs=[gr.Image(label="Image Segmentée"), gr.Textbox(label="Diagnostic")],  
    title="AppleHealth",
    description="Téléchargez une image de votre plante pour effectuer une analyse de segmentation et obtenir un diagnostic de sa santé.",
    live=True  
)

# Lancer l'application Gradio
iface.launch()
