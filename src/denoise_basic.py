import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output folder
os.makedirs("outputs/reverse_process", exist_ok=True)

# Load the most noisy image from forward process
img = cv2.imread("outputs/forward_process/step_4.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Normalize
img = img.astype(np.float32) / 255.0

def denoise_step(image, step):
    """
    Apply stronger denoising as step increases.
    """
    img_uint8 = (image * 255).astype(np.uint8)

    if step == 1:
        denoised = cv2.GaussianBlur(img_uint8, (3, 3), 0)
    elif step == 2:
        denoised = cv2.GaussianBlur(img_uint8, (5, 5), 0)
    elif step == 3:
        denoised = cv2.medianBlur(img_uint8, 5)
    elif step == 4:
        denoised = cv2.bilateralFilter(img_uint8, 9, 75, 75)
    else:
        denoised = img_uint8

    return denoised.astype(np.float32) / 255.0

images = [img]
titles = ["Input Noisy Image"]

current = img.copy()

for step in range(1, 5):
    current = denoise_step(current, step)
    images.append(current)
    titles.append(f"Denoised Step {step}")

    save_img = (current * 255).astype(np.uint8)
    save_img_bgr = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"outputs/reverse_process/denoised_step_{step}.png", save_img_bgr)

# Plot results
plt.figure(figsize=(16, 4))
for i, im in enumerate(images):
    plt.subplot(1, len(images), i + 1)
    plt.imshow(im)
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()