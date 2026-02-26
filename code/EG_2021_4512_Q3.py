import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Helper function to save outputs
def save_img(title, img, fname):
    plt.figure(figsize=(5,5))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    
    out_path = os.path.join("../outputs", fname)
    plt.savefig(out_path, bbox_inches='tight')
    print("[SAVED] ", out_path)
    plt.close()

if __name__ == "__main__":

    # Load Image 3 (grayscale)
    img = cv2.imread("../images/Image_3.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image_3.jpg not found.")
        exit()

    # Kernel sizes
    kernels = [3, 5, 11, 15]

    # ----- Part 1: Gaussian filtering for different kernels -----
    for k in kernels:
        blurred = cv2.GaussianBlur(img, (k, k), sigmaX=0)
        save_img(f"Gaussian {k}x{k}", blurred, f"Q3_gauss_{k}x{k}.png")

    # ----- Part 2: Sigma variation (use 11x11 kernel) -----
    sigmas = [0.5, 1, 2, 5]

    for sigma in sigmas:
        blurred_sigma = cv2.GaussianBlur(img, (11, 11), sigmaX=sigma)
        save_img(f"Gaussian 11x11, sigma={sigma}", blurred_sigma, f"Q3_sigma_{sigma}.png")