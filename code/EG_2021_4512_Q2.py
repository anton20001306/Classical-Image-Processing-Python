import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Helper function
def save_img(title, img, fname):
    plt.figure(figsize=(5,5))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    
    out_path = os.path.join("../outputs", fname)
    plt.savefig(out_path, bbox_inches='tight')
    print("[SAVED] ", out_path)
    plt.close()


# Salt & Pepper Noise function
def add_salt_pepper_noise(image, amount):
    noisy = image.copy()
    num_pixels = int(amount * image.size)

    # S A L T
    y_coords = np.random.randint(0, image.shape[0], num_pixels)
    x_coords = np.random.randint(0, image.shape[1], num_pixels)
    noisy[y_coords, x_coords] = 255

    # P E P P E R
    y_coords = np.random.randint(0, image.shape[0], num_pixels)
    x_coords = np.random.randint(0, image.shape[1], num_pixels)
    noisy[y_coords, x_coords] = 0

    return noisy


if __name__ == "__main__":

    # Load image 2
    img = cv2.imread("../images/Image_2.jpg", cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Image_2.jpg not found")
        exit()

    # Noise levels
    noise_levels = [0.10, 0.20]

    # Kernel sizes
    kernels = [3, 5, 11]

    for noise in noise_levels:

        noisy_img = add_salt_pepper_noise(img, noise)
        save_img(
            f"Salt & Pepper Noise ({int(noise*100)}%)",
            noisy_img,
            f"Q2_noise_{int(noise*100)}.png"
        )

        for k in kernels:
            filtered = cv2.medianBlur(noisy_img, k)

            save_img(
                f"Median Filter {k}x{k} ({int(noise*100)}% noise)",
                filtered,
                f"Q2_median_{k}_{int(noise*100)}.png"
            )