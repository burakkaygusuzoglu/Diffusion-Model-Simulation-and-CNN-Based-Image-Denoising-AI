import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(__file__))
from cnn_denoiser import SimpleDenoiseCNN

print("Program started")

# Create output directories at the start
os.makedirs("outputs/cnn_results", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("Loading image...")
img = cv2.imread("data/apple.jpg")
if img is None:
    raise FileNotFoundError("data/apple.jpg not found")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (64, 64))
img = img.astype(np.float32) / 255.0

def add_noise(image, noise_level=0.2):
    noise = np.random.normal(0, noise_level, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 1)

print("Creating dataset...")
X_train = []
Y_train = []

for _ in range(50):
    noisy_img = add_noise(img, noise_level=np.random.uniform(0.1, 0.4))
    X_train.append(noisy_img)
    Y_train.append(img)

X_train = np.array(X_train, dtype=np.float32)
Y_train = np.array(Y_train, dtype=np.float32)

X_train = torch.tensor(X_train).permute(0, 3, 1, 2)
Y_train = torch.tensor(Y_train).permute(0, 3, 1, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = SimpleDenoiseCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train = X_train.to(device)
Y_train = Y_train.to(device)

epochs = 10
loss_history = []

print("Starting training...")

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.6f}")

print("Training finished")

torch.save(model.state_dict(), "models/cnn_denoiser.pth")

print("Running test...")
test_noisy = add_noise(img, noise_level=0.3)
test_input = torch.tensor(test_noisy, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    denoised_output = model(test_input).cpu().squeeze(0).permute(1, 2, 0).numpy()

denoised_output = np.clip(denoised_output, 0, 1)

cv2.imwrite("outputs/cnn_results/test_noisy.png", cv2.cvtColor((test_noisy * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
cv2.imwrite("outputs/cnn_results/test_denoised.png", cv2.cvtColor((denoised_output * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
cv2.imwrite("outputs/cnn_results/ground_truth.png", cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

print("Images saved in outputs/cnn_results/")

# Save plots instead of blocking with plt.show()
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(test_noisy)
plt.title("Noisy Input")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(denoised_output)
plt.title("CNN Output")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img)
plt.title("Ground Truth")
plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/cnn_results/comparison.png", dpi=100, bbox_inches='tight')
print("Comparison plot saved to outputs/cnn_results/comparison.png")
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("outputs/cnn_results/loss_history.png", dpi=100, bbox_inches='tight')
print("Loss plot saved to outputs/cnn_results/loss_history.png")
plt.close()