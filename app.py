from flask import Flask, render_template, request, redirect, url_for
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 6
classes = ['Acral_Lentiginous_Melanoma', 'Healthy_Nail', 'Onychogryphosis', 'blue_finger', 'clubbing', 'pitting']

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model.load_state_dict(torch.load("nail_classifier_resnet18.pth", map_location=device))
model.eval()
model = model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def generate_gradcam(model, img_tensor, target_class, device):
    model.eval()
    gradients, activations = [], []

    def save_gradient(grad):
        gradients.append(grad)

    target_layer = model.layer4[-1]
    target_layer.register_forward_hook(lambda m, i, o: activations.append(o))
    target_layer.register_full_backward_hook(lambda m, grad_in, grad_out: save_gradient(grad_out[0]))

    output = model(img_tensor)
    model.zero_grad()
    class_loss = output[0, target_class]
    class_loss.backward()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam / cam.max() if cam.max() != 0 else cam
    return cam

def calculate_severity(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Unknown"
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    h, w = img_gray.shape
    total = h * w

    color_std = np.std(img_gray) / 255.0

    edges = cv2.Canny(img_gray, 100, 200)
    edge_pixels = np.count_nonzero(edges)
    edge_density = edge_pixels / total

    _, otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = max((cv2.contourArea(c) for c in contours), default=0.0)
    contour_area_ratio = largest_area / total if total > 0 else 0.0

    severity_score = 0.6 * color_std + 0.25 * edge_density + 0.15 * (1.0 - contour_area_ratio)

    if severity_score < 0.15:
        return "Mild"
    elif severity_score < 0.35:
        return "Moderate"
    else:
        return "Severe"

def map_to_systemic_disease(predicted_class):
    mapping = {
        "Acral_Lentiginous_Melanoma": "Possible link to melanoma (skin cancer). Urgent medical evaluation recommended.",
        "Onychogryphosis": "May be associated with poor circulation, psoriasis, or trauma-related deformity.",
        "blue_finger": "Can indicate hypoxia, heart/lung issues, or Raynaud’s phenomenon.",
        "clubbing": "Often linked to chronic lung disease, heart disease, or liver cirrhosis.",
        "pitting": "Common in psoriasis or autoimmune disorders.",
        "Healthy_Nail": "No systemic disease indication — normal condition."
    }
    return mapping.get(predicted_class, "No known systemic mapping available.")

def format_label(name: str) -> str:
    return name.replace('_', ' ').replace('-', ' ').title()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)

        img_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(img_path)

        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)[0]
            pred_class = torch.argmax(probs).item()
            original_label = classes[pred_class]
            formatted_label = format_label(original_label)

        cam = generate_gradcam(model, img_tensor, pred_class, device)
        img_cv = cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = np.float32(heatmap) * 0.4 + np.float32(img_cv)
        overlay = overlay / np.max(overlay)
        cam_image = cv2.cvtColor(np.uint8(255 * overlay), cv2.COLOR_BGR2RGB)

        _, buffer = cv2.imencode('.png', cam_image)
        cam_base64 = base64.b64encode(buffer).decode('utf-8')

        severity = calculate_severity(img_path)
        if original_label == "Healthy_Nail":
            severity = "Mild"
        systemic_info = map_to_systemic_disease(original_label)

        prob_dict = {format_label(classes[i]): round(probs[i].item() * 100, 2) for i in range(len(classes))}

        return render_template("result.html",
                               label=formatted_label,
                               probs=prob_dict,
                               cam_image=cam_base64,
                               severity=severity,
                               systemic=systemic_info)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
