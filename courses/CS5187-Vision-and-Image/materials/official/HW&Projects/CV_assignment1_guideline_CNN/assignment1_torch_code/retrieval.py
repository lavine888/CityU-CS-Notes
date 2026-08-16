# Retrieve the most similar images by measuring the similarity between features.
import numpy as np
import os
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt

# Measure the similarity scores between query feature and gallery features.
# You could also use other metrics to measure the similarity scores between features.
def similarity(query_feat, gallery_feat):
    sim = cosine_similarity(query_feat, gallery_feat)
    sim = np.squeeze(sim)
    return sim

def retrival_idx(query_path, gallery_dir):
    query_feat = np.load(query_path)
    dict = {}
    for gallery_file in os.listdir(gallery_dir):
        gallery_feat = np.load(os.path.join(gallery_dir, gallery_file))
        gallery_idx = gallery_file.split('.')[0] + '.jpg'
        sim = similarity(query_feat, gallery_feat)
        dict[gallery_idx] = sim
    sorted_dict = sorted(dict.items(), key=lambda item: item[1]) # Sort the similarity score
    best_five = sorted_dict[-5:] # Get the best five retrived images
    return best_five

def visulization(retrived, query):
    plt.subplot(2, 3, 1)
    plt.title('query')
    query_img = cv2.imread(query)
    img_rgb_rgb = query_img[:,:,::-1]
    plt.imshow(img_rgb_rgb)
    for i in range(5):
        img_path = './data/gallery/' + retrived[i][0]
        img = cv2.imread(img_path)
        img_rgb = img[:,:,::-1]
        plt.subplot(2, 3, i+2)
        plt.title(retrived[i][1])
        plt.imshow(img_rgb)
    plt.show()

if __name__ == '__main__':
    query_path = './data/query_feat/query_feats.npy'
    gallery_dir = './data/gallery_feature/'
    best_five = retrival_idx(query_path, gallery_dir) # retrieve top 5 matching images in the gallery.
    print(best_five)
    best_five.reverse()
    query_path = './data/query/query.jpg'
    visulization(best_five, query_path) # Visualize the retrieval results

