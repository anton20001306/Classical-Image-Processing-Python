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

# Our manual Gaussian pyramid function (same as Q4)
def manual_gaussian_pyr(img, levels=3):
    pyramid = [img]
    current = img.copy()
    for _ in range(levels):
        # Simulate pyrDown: blur + downsample
        blur = cv2.GaussianBlur(current, (5,5), 0)
        down = blur[::2, ::2]   # simple downsampling
        pyramid.append(down)
        current = down
    return pyramid

if __name__ == "__main__":

    # Load Image 3 (grayscale)
    img = cv2.imread("../images/Image_3.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image_3.jpg not found!")
        exit()

    # ------------------------------
    # Manual Gaussian Pyramid
    # ------------------------------
    manual_pyr = manual_gaussian_pyr(img, levels=3)

    # Save manual levels
    for i, level_img in enumerate(manual_pyr):
        save_img(f"Manual Gaussian Level {i}", level_img, f"Q5_manual_lvl{i}.png")

    # ------------------------------
    # OpenCV Gaussian Pyramid
    # ------------------------------
    cv_pyr = [img]
    current = img.copy()

    for i in range(3):
        current = cv2.pyrDown(current)
        cv_pyr.append(current)
        save_img(f"OpenCV pyrDown Level {i+1}", current, f"Q5_cv_lvl{i+1}.png")

    # ------------------------------
    # Differences (Absolute)
    # ------------------------------
    for i in range(1, 4):
        # Resize manual to match cv_pyr size
        manual_resized = cv2.resize(manual_pyr[i], (cv_pyr[i].shape[1], cv_pyr[i].shape[0]))
        
        diff = cv2.absdiff(manual_resized, cv_pyr[i])
        
        save_img(
            f"Difference Level {i}",
            diff,
            f"Q5_diff_lvl{i}.png"
        )