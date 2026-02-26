import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def save_img(title, img, fname, cmap='gray'):
    plt.figure(figsize=(5,5))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    path = os.path.join("../outputs", fname)
    plt.savefig(path, bbox_inches='tight')
    print("[SAVED]", path)
    plt.close()

if __name__ == "__main__":

    # Load MRI image
    img = cv2.imread("../images/Image_5.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image_5.jpg not found!")
        exit()

    save_img("Original MRI", img, "Q9_original.png")

    # 1. Gaussian Denoising
    denoised = cv2.GaussianBlur(img, (5,5), 0)
    save_img("Gaussian Denoised", denoised, "Q9_denoised.png")

    # 2. Histogram Equalization
    hist_eq = cv2.equalizeHist(img)
    save_img("Histogram Equalized", hist_eq, "Q9_histeq.png")

    # 3. CLAHE (Adaptive Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(img)
    save_img("CLAHE Enhanced", clahe_img, "Q9_CLAHE.png")

    # 4. Sharpening (Unsharp Mask)
    blurred = cv2.GaussianBlur(img, (9,9), 10)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    save_img("Sharpened Image", sharpened, "Q9_sharpened.png")

    # 5. Combine results for comparison (optional)
    comparison = np.hstack((img, clahe_img, sharpened))
    save_img("Comparison: Original - CLAHE - Sharpened", comparison, "Q9_compare.png", cmap='gray')