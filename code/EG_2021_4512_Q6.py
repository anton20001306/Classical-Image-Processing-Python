import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os

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

    # Load Image 3
    img = cv2.imread("../images/Image_3.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image_3.jpg not found!")
        exit()

    # ----------------------------
    # 1-level DWT
    # ----------------------------
    wavelet = 'haar'
    coeffs2 = pywt.dwt2(img, wavelet)
    LL, (LH, HL, HH) = coeffs2

    # Save each sub-band
    save_img("LL - Approximation", LL, "Q6_LL.png")
    save_img("LH - Horizontal", LH, "Q6_LH.png")
    save_img("HL - Vertical", HL, "Q6_HL.png")
    save_img("HH - Diagonal", HH, "Q6_HH.png")

    # ----------------------------
    # Inverse DWT (Reconstruction)
    # ----------------------------
    reconstructed = pywt.idwt2((LL, (LH, HL, HH)), wavelet)
    reconstructed = np.uint8(np.clip(reconstructed, 0, 255))

    save_img("Reconstructed Image", reconstructed, "Q6_reconstructed.png")

    # ----------------------------
    # Difference between original and reconstructed
    # ----------------------------
    diff = cv2.absdiff(img, reconstructed)
    save_img("Difference Image", diff, "Q6_diff.png")