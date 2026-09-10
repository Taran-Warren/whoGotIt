import warnings
warnings.filterwarnings("ignore")
import torch
import cv2
import open_clip
from PIL import Image
from mobileclip.modules.common.mobileone import reparameterize_model
import time



model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP2-S2', pretrained=r'C:\Users\Warren\.cache\huggingface\hub\models--apple--MobileCLIP2-S2\snapshots\72424e7025436db18f15c3eff6ee8c7c15ad4481\mobileclip2_s2.pt')
#model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP2-S0', pretrained=r'C:\Users\Warren\.cache\huggingface\hub\models--apple--MobileCLIP2-S0\snapshots\3136ea51c8ed56b9f9abfab04cb816735aaad6cb\mobileclip2_s0.pt')
model = reparameterize_model(model)
model.eval()

def encode_image(image_path):
    img = preprocess(Image.open(image_path)).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(img)
    return embedding

def resize_image(input_path,output_path, size = 700):
    img = Image.open(input_path)
    img.thumbnail((size,size))
    img.save(output_path)


def match_orb(image_path1,image_path2):
    img1 = cv2.imread(image_path1,cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2,cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    kp1,des1 = orb.detectAndCompute(img1,None)
    kp2,des2 = orb.detectAndCompute(img2,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    matches = bf.match(des1,des2)
    return len(matches)

