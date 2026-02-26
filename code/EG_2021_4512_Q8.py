import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

def save_img(title, img, fname):
    plt.figure(figsize=(6,5))
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    path = os.path.join("../outputs", fname)
    plt.savefig(path, bbox_inches='tight')
    print("[SAVED]", path)
    plt.close()

if __name__ == "__main__":
    # Load image
    img = cv2.imread("../images/Image_4.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error loading Image_4.jpg")
        exit()

    save_img("Original CT", img, "Q8_original.png")

    # Preprocessing
    blur = cv2.GaussianBlur(img, (7,7), 0)

    # Reshape for clustering
    pixel_vals = blur.reshape((-1,1)).astype(np.float32)

    # K-means clustering (K=5)
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(pixel_vals)
    labels = kmeans.labels_.reshape(img.shape)

    # Color each cluster
    colors = np.random.randint(0,255,(k,3))
    segmented = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for i in range(k):
        segmented[labels == i] = colors[i]

    save_img("K-means Tissue Segmentation", segmented, "Q8_kmeans.png")

    # Connected components within each cluster
    organ_map = np.zeros_like(segmented)

    for i in range(k):
        mask = (labels == i).astype(np.uint8) * 255
        num, labs = cv2.connectedComponents(mask)

        # Color each connected region
        for cc in range(1, num):
            organ_map[labs == cc] = np.random.randint(0,255,3)

    save_img("Final Multi-Organ Segmentation", organ_map, "Q8_segmented.png")