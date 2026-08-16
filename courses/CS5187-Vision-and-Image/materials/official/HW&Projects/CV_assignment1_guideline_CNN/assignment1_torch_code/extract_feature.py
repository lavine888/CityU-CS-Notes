#HOW TO INSTALL ANNACONDA: https://www.youtube.com/watch?v=YJC6ldI3hWk
#WHAT IS IMAGENET DATABASE: https://www.youtube.com/watch?v=gogV2wKKF_8
import torch
import cv2
import os
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

#crop the instance region. For the images containing two instances, you need to crop both of them.
def query_crop(query_path, txt_path, save_path):
    query_img = cv2.imread(query_path)
    query_img = query_img[:,:,::-1] #bgr2rgb
    txt = np.loadtxt(txt_path)     #load the coordinates of the bounding box
    crop = query_img[int(txt[1]):int(txt[1] + txt[3]), int(txt[0]):int(txt[0] + txt[2]), :] #crop the instance region
    cv2.imwrite(save_path, crop[:,:,::-1])  #save the cropped region
    return crop

def vgg_11_extraction(img, featsave_path):
    resnet_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    img_transform = resnet_transform(img) #normalize the input image and transform it to tensor.
    img_transform = torch.unsqueeze(img_transform, 0) #Set batchsize as 1. You can enlarge the batchsize to accelerate.

    # initialize the weights pretrained on the ImageNet dataset, you can also use other backbones (e.g. ResNet, XceptionNet, AlexNet, ...)
    # and extract features from more than one layer.
    vgg11 = models.vgg11(pretrained=True)
    vgg11_feat_extractor = vgg11.features #define the feature extractor
    vgg11_feat_extractor.eval()  #set the mode as evaluation
    feats = vgg11(img_transform) # extract feature
    feats_np = feats.cpu().detach().numpy() # convert tensor to numpy
    np.save(featsave_path, feats_np) # save the feature

# Note that I feed the whole image into the pretrained vgg11 model to extract the feature, which will lead to a poor retrieval performance.
# To extract more fine-grained features, you could preprocess the gallery images by cropping them using windows with different sizes and shapes.
# Hint: opencv provides some off-the-shelf tools for image segmentation.
def feat_extractor_gallery(gallery_dir, feat_savedir):
    for img_file in tqdm(os.listdir(gallery_dir)):
        img = cv2.imread(os.path.join(gallery_dir, img_file))
        img = img[:,:,::-1] #bgr2rgb
        img_resize = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC) # resize the image
        featsave_path = os.path.join(feat_savedir, img_file.split('.')[0]+'.npy')
        vgg_11_extraction(img_resize, featsave_path)

# Extract the query feature
def feat_extractor_query():
    query_path = './data/query/query.jpg'
    txt_path = './data/query_txt/query.txt'
    save_path = './data/cropped_query/query.jpg'
    featsave_path = './data/query_feat/query_feats.npy'
    crop = query_crop(query_path, txt_path, save_path)
    crop_resize = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
    vgg_11_extraction(crop_resize, featsave_path)

def main():
    feat_extractor_query()
    gallery_dir = './data/gallery/'
    feat_savedir = './data/gallery_feature/'
    feat_extractor_gallery(gallery_dir, feat_savedir)

if __name__=='__main__':
    main()