import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# -------- HELPER FUNCTION ----------
def show_and_save(title, img, fname):
    plt.figure(figsize=(5,5))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    # Save output
    out_path = os.path.join("../outputs", fname)
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()

# -------- MAIN PROGRAM ----------
if __name__ == "__main__":
    # Load image in grayscale
    img = cv2.imread("../images/Image_1.jpg", cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: image_1 not found in images folder")
        exit()

    # Kernel sizes
    kernels = [3, 5, 11, 15]

    for k in kernels:
        # Create averaging filter
        kernel = np.ones((k, k), np.float32) / (k * k)

        # Apply filter
        filtered = cv2.filter2D(img, -1, kernel)

        # Display + save
        show_and_save(
            title=f"Average Filter {k}x{k}",
            img=filtered,
            fname=f"Q1_avg_{k}x{k}.png"
        )