import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Helper to save images
def save_img(title, img, fname):
    plt.figure(figsize=(5,5))
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')

    out_dir = "../outputs"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    save_path = os.path.join(out_dir, fname)
    plt.savefig(save_path, bbox_inches='tight')
    print("[SAVED]", save_path)
    plt.close()


# Function to add salt & pepper noise
def add_salt_pepper(image, amount):
    noisy = image.copy()
    prob = amount

    # Salt noise (white pixels)
    num_salt = int(prob * image.size * 0.5)
    coords_salt = (
        np.random.randint(0, image.shape[0], num_salt),
        np.random.randint(0, image.shape[1], num_salt)
    )
    noisy[coords_salt] = 255

    # Pepper noise (black pixels)
    num_pepper = int(prob * image.size * 0.5)
    coords_pepper = (
        np.random.randint(0, image.shape[0], num_pepper),
        np.random.randint(0, image.shape[1], num_pepper)
    )
    noisy[coords_pepper] = 0

    return noisy


if __name__ == "__main__":
    # Load Image 2 in grayscale
    img = cv2.imread("../images/Image_2.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image_2.jpg not found!")
        exit()

    # (a) Add noise
    noise10 = add_salt_pepper(img, 0.10)
    noise20 = add_salt_pepper(img, 0.20)

    save_img("Salt & Pepper Noise (10%)", noise10, "Q2_noise_10.png")
    save_img("Salt & Pepper Noise (20%)", noise20, "Q2_noise_20.png")

    # (b) Median filters
    med3 = cv2.medianBlur(noise20, 3)
    med5 = cv2.medianBlur(noise20, 5)
    med11 = cv2.medianBlur(noise20, 11)

    save_img("Median Filter 3x3", med3, "Q2_med_3x3.png")
    save_img("Median Filter 5x5", med5, "Q2_med_5x5.png")
    save_img("Median Filter 11x11", med11, "Q2_med_11x11.png")