import cv2
import numpy as np
import pywt
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

    # Load host image
    host = cv2.imread("../images/Image_3.jpg", cv2.IMREAD_GRAYSCALE)
    if host is None:
        print("Error: host image not found!")
        exit()

    # Load watermark image
    watermark = cv2.imread("../images/watermark.png", cv2.IMREAD_GRAYSCALE)
    if watermark is None:
        print("Error: watermark image not found!")
        exit()

    # DWT on host image
    LL, (LH, HL, HH) = pywt.dwt2(host, 'haar')

    # Resize watermark to match LL
    watermark_resized = cv2.resize(watermark, (LL.shape[1], LL.shape[0]))

    # Embedding strength
    alpha = 0.05

    # Embed watermark in LL band
    LL_w = LL + alpha * watermark_resized

    # Inverse DWT → watermarked image
    watermarked = pywt.idwt2((LL_w, (LH, HL, HH)), 'haar')
    watermarked = np.uint8(np.clip(watermarked, 0, 255))

    # Save outputs
    save_img("Watermark resized", watermark_resized, "Q7_watermark_resized.png")
    save_img("Watermarked Image", watermarked, "Q7_watermarked.png")

    # ----------------------------
    # Watermark Extraction
    # ----------------------------

    # Apply DWT on watermarked image
    LL_wm, (LH2, HL2, HH2) = pywt.dwt2(watermarked, 'haar')

    # Extract watermark
    watermark_recovered = (LL_wm - LL) / alpha
    watermark_recovered = np.uint8(np.clip(watermark_recovered, 0, 255))

    # Save
    save_img("Recovered Watermark", watermark_recovered, "Q7_recovered.png")

    # Difference image
    diff = cv2.absdiff(watermark_resized, watermark_recovered)
    save_img("Difference", diff, "Q7_difference.png")