import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output folder
os.makedirs("outputs/forward_process", exist_ok=True)

# Load image
img = cv2.imread("data/apple.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))
img = img.astype(np.float32) / 255.0

def add_noise(image, noise_level):
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image

noise_levels = [0.0, 0.1, 0.2, 0.35, 0.5]
images = []

for i, level in enumerate(noise_levels):
    noisy = add_noise(img, level)
    images.append(noisy)
    save_img = (noisy * 255).astype(np.uint8)
    cv2.imwrite(f"outputs/forward_process/step_{i}.png", cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))

plt.figure(figsize=(15, 4))
for i, im in enumerate(images):
    plt.subplot(1, len(images), i + 1)
    plt.imshow(im)
    plt.title(f"Step {i}\nNoise={noise_levels[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()