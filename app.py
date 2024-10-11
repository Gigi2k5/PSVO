import gradio as gr
from PIL import Image
import numpy as np
import cv2
from skimage import io
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from scipy import ndimage
from skimage import morphology
import os

# Fonctions de traitement d'image
def load_image(image):
    return image

def apply_negative(image):
    img_np = np.array(image)
    negative = 255 - img_np
    return Image.fromarray(negative)

def binarize_image(image, threshold):
    img_np = np.array(image.convert('L'))
    _, binary = cv2.threshold(img_np, threshold, 255, cv2.THRESH_BINARY)
    return Image.fromarray(binary)

def resize_image(image, width, height):
    return image.resize((int(width), int(height)))

def rotate_image(image, angle):
    return image.rotate(angle)

# Ajoutez d'autres fonctions
"""Nous avons défini une fonction pour afficher l'Histogramme. Cette fonction
vérifie si l'image est en blanc et noir ou s'il s'agit d'une image en couleur. 
S'il s'agit d'une image blanc et noir elle applique la première condition et 
sort un seul graphe sinon elle applique le else et sort trois graphes correspondant
à chaque canaux de couleur
"""
def histogramOpenCV(image):
    img_np = np.array(image)
    if img_np.dtype != 'uint8':
        img_np = (img_np * 255).astype('uint8')
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np, cmap='gray')
    plt.title('Image')
    
    plt.subplot(1, 2, 2)
    plt.title('Histogram')
    
    if len(img_np.shape) == 2:
        histr = cv2.calcHist([img_np], [0], None, [256], [0, 256])
        plt.plot(histr, color='blue')
    else:
        rgbcolors = ['red', 'green', 'blue']
        for i, col in enumerate(rgbcolors):
            histr = cv2.calcHist([img_np], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
    
    plt.tight_layout()
    plt.savefig('histogram_temp.png')
    plt.close()
    return Image.open('histogram_temp.png')
# Nous réalisons un seuillage otsu grace à cette fonction

def thresholdOtsuDisplay(image):
    img_np = np.array(image.convert('L'))
    thresh = threshold_otsu(img_np)
    binary = img_np > thresh
    binary_image = (binary * 255).astype(np.uint8)

    return Image.fromarray(binary_image)
# Cette fonction nous permet d'extraire les contours d'une image
def extract(image): 
    img_np = np.array(image.convert('L'))
    kernel_contour = np.array([[0, 1, 0],
                                [1, -4, 1], 
                                [0, 1, 0]])
    imgconvol = ndimage.convolve(img_np, kernel_contour, mode='reflect')
    
    return Image.fromarray(imgconvol)
# Nous réalisons ici deux transformations morphologiques notamment 
#l'érosion et la dilatation
def erosion(image): 
    img_np = np.array(image.convert('L'))
    result_erosion = morphology.binary_erosion(img_np, morphology.disk(1))
    
    result_erosion_img = (result_erosion * 255).astype(np.uint8)
    return Image.fromarray(result_erosion_img)

def dilatation(image): 
    img_np = np.array(image.convert('L'))
    dilation = morphology.binary_dilation(img_np, morphology.disk(1))
    
    dilation_img = (dilation * 255).astype(np.uint8)
    return Image.fromarray(dilation_img)

# Sauvegarde de l'image
def save(image, filename):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Vérification de l'extension et atttribution d'une extension par défaut 
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
        filename += '.png'  

    image.save(filename)
    return f"Image saved as {filename}"

# Interface Gradio
def image_processing(image, operation, threshold=128, width=100, height=100, angle=0):
    if operation == "Négatif":
        return apply_negative(image)
    elif operation == "Binarisation":
        return binarize_image(image, threshold)
    elif operation == "Redimensionner":
        return resize_image(image, width, height)
    elif operation == "Rotation":
        return rotate_image(image, angle)
    elif operation == "Histogramme":
        return histogramOpenCV(image) 
    elif operation == "Seuil Otsu":
        return thresholdOtsuDisplay(image)
    elif operation == "Extraction de contours":
        return extract(image)
    elif operation == "Erosion":
        return erosion(image)
    elif operation == "Dilatation":
        return dilatation(image)
    
    return image

# Fonction pour mettre à jour dynamiquement la visibilité des éléments
# Nous avons créer cette fonction pour permettre à l'utilisateur de 
# modifier certaines variables comme le height, le width,...
def update_ui(operation):
    if operation == "Binarisation":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    elif operation == "Redimensionner":
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
    elif operation == "Rotation":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

# Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("## Projet de Traitement d'Image")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Charger Image")
        operation = gr.Radio(["Négatif", "Binarisation", "Redimensionner", "Rotation", "Histogramme", "Seuil Otsu", "Extraction de contours",  "Erosion", "Dilatation"], label="Opération")
        threshold = gr.Slider(0, 255, 128, label="Seuil de binarisation", visible=False)
        width = gr.Number(value=100, label="Largeur", visible=False)
        height = gr.Number(value=100, label="Hauteur", visible=False)
        angle = gr.Number(value=0, label="Angle de Rotation", visible=False)
        filename = gr.Textbox(label="Nom du fichier pour sauvegarder", placeholder="image_modifiee.png")

    image_output = gr.Image(label="Image Modifiée")

    submit_button = gr.Button("Appliquer")
    submit_button.click(image_processing, inputs=[image_input, operation, threshold, width, height, angle], outputs=image_output)
    operation.change(update_ui, inputs=[operation], outputs=[threshold, width, height, angle])

    save_button = gr.Button("Sauvegarder")
    save_button.click(save, inputs=[image_output, filename], outputs=[gr.Textbox()])

# Lancer l'application Gradio
demo.launch()
