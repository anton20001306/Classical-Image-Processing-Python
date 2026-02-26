import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def save_img(title, img, fname):
    plt.figure(figsize=(5,5))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    out_path = os.path.join("../outputs", fname)
    plt.savefig(out_path, bbox_inches='tight')
    print("[SAVED]", out_path)
    plt.close()


if __name__ == "__main__":
    
    # Load image 3
    img = cv2.imread("../images/Image_3.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image_3.jpg not found!")
        exit()

    # -------------------------------------------------
    # PART 1: Gaussian Pyramid (3-level)
    # -------------------------------------------------

    gaussian_pyramid = [img]
    current = img.copy()

    for level in range(1, 4):       # 3 levels: Level1, Level2, Level3
        current = cv2.pyrDown(current)
        gaussian_pyramid.append(current)
        save_img(f"Gaussian Pyramid Level {level}", current, f"Q4_gauss_level{level}.png")

    # Save original as Level 0
    save_img("Gaussian Pyramid Level 0 (Original)", img, "Q4_gauss_level0.png")

    # -------------------------------------------------
    # PART 2: Laplacian Pyramid (3-level)
    # -------------------------------------------------

    laplacian_pyramid = []

    for level in range(1,4):  # Laplacian levels from Gaussian
        size = (gaussian_pyramid[level-1].shape[1], gaussian_pyramid[level-1].shape[0])
        gaussian_up = cv2.pyrUp(gaussian_pyramid[level], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[level-1], gaussian_up)
        laplacian_pyramid.append(laplacian)
        
        save_img(
            f"Laplacian Pyramid Level {level}",
            laplacian,
            f"Q4_lap_level{level}.png"
        )