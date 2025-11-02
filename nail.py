import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device in use: {device}")

data_dir = "data"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "validation")

batch_size = 16
num_epochs = 5
learning_rate = 0.0001
num_classes = 6

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

classes = train_dataset.classes
print("🧩 Classes found:", classes)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("""
Nail Disease Classifier
-----------------------
1️⃣ Train Model
2️⃣ Validate Model
3️⃣ Predict Single Image
""")
choice = input("Enter choice (1/2/3): ").strip()

if choice == "1":
    print("\n🚀 Starting training...\n")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Loss: {running_loss/len(train_loader):.4f} | "
              f"Accuracy: {train_acc:.2f}%")

    torch.save(model.state_dict(), "nail_classifier_resnet18.pth")
    print("\n💾 Model saved as 'nail_classifier_resnet18.pth'")

elif choice == "2":
    model.load_state_dict(torch.load("nail_classifier_resnet18.pth", map_location=device))
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"\n✅ Validation Accuracy: {val_acc:.2f}%")

elif choice == "3":
    model.load_state_dict(torch.load("nail_classifier_resnet18.pth", map_location=device))
    model.eval()

    img_path = input("\nEnter image path for prediction: ").strip()
    if not os.path.exists(img_path):
        print("❌ File not found.")
    else:
        img = Image.open(img_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)[0]
            pred_class = torch.argmax(probs).item()
            pred_label = classes[pred_class]

        print(f"\n🔮 Predicted Class: {pred_label}")
        print("\nPrediction Percentages:")
        for i, prob in enumerate(probs):
            print(f"{classes[i]}: {prob.item()*100:.2f}%")

        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Prediction: {pred_label}")
        plt.show()

    def generate_gradcam(model, img_tensor, target_class, device):
        model.eval()
        gradients = []
        activations = []

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
        max_val = cam.max()
        if max_val != 0:
            cam = cam / max_val
        return cam

    cam = generate_gradcam(model, img_tensor, pred_class, device)
    img_cv = cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = np.float32(heatmap) * 0.4 + np.float32(img_cv)
    overlay = overlay / np.max(overlay)

    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(np.uint8(255 * overlay), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Grad-CAM: {pred_label}", fontsize=16)
    plt.show()

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

    severity = calculate_severity(img_path)
    if pred_label == "Healthy_Nail":
        severity = "Mild"
    systemic_info = map_to_systemic_disease(pred_label)

    print(f"\n🩹 Severity Level: {severity}")
    print(f"🫀 Systemic Mapping: {systemic_info}")

else:
    print("❌ Invalid choice. Please restart and enter 1, 2, or 3.")
