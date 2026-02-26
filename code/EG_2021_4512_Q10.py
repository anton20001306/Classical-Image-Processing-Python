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

    # Load geometric shapes image
    img = cv2.imread("../images/Image_6.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image_6.jpg not found!")
        exit()

    save_img("Original Image", img, "Q10_original.png")

    # Binary threshold
    _, binary = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    save_img("Binary Image", binary, "Q10_binary.png")

    # Structuring element
    kernel = np.ones((5,5), np.uint8)

    # 1. Erosion
    erosion = cv2.erode(binary, kernel, iterations=1)
    save_img("Erosion", erosion, "Q10_erosion.png")

    # 2. Dilation
    dilation = cv2.dilate(binary, kernel, iterations=1)
    save_img("Dilation", dilation, "Q10_dilation.png")

    # 3. Opening (erosion then dilation)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    save_img("Opening", opening, "Q10_opening.png")

    # 4. Closing (dilation then erosion)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    save_img("Closing", closing, "Q10_closing.png")

    # 5. Edge extraction by morphological gradient
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    save_img("Morphological Gradient", gradient, "Q10_gradient.png")

    # 6. Connected components (shape extraction)
    num_labels, labels = cv2.connectedComponents(binary)
    colors = np.random.randint(0,255,(num_labels,3))
    segmented = colors[labels]

    save_img("Segmented Shapes", segmented, "Q10_segmented.png", cmap=None)