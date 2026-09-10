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
image1 = preprocess(Image.open("test_1a.jpg")).unsqueeze(0)
image2 = preprocess(Image.open("test_1b.jpg")).unsqueeze(0)
image3 = preprocess(Image.open("test_2a.jpg")).unsqueeze(0)
image4 = preprocess(Image.open("test_3a.jpg")).unsqueeze(0)
image5 = preprocess(Image.open("test_1c.png")).unsqueeze(0)
image6 = preprocess(Image.open("test_2c.png")).unsqueeze(0)
image7 = preprocess(Image.open("test_1d.png")).unsqueeze(0)

with torch.no_grad():
    embedding1 = model.encode_image(image1)
    embedding2 = model.encode_image(image2)
    embedding3 = model.encode_image(image3)
    embedding4 = model.encode_image(image4)
    embedding5 = model.encode_image(image5)
    embedding6 = model.encode_image(image6)
    embedding7 = model.encode_image(image7)

similarity_same = torch.nn.functional.cosine_similarity(embedding5, embedding5)
similarity_diff = torch.nn.functional.cosine_similarity(embedding5, embedding4)
print("Same object similarity:", similarity_same.item())
print("Different object similarity:", similarity_diff.item())


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
resize_image("test_1c.png","test_1cr.png")
resize_image("test_1d.png","test_1dr.png")
resize_image("test_2c.png","test_2cr.png")
resize_image("test_3b.png","test_3br.png")



start = time.time()
score = match_orb("test_1cr.png","test_1dr.png")
print (score)
score = match_orb("test_2cr.png","test_1cr.png")
print (score)
score = match_orb("test_1cr.png","test_3br.png")
print (score)
end = time.time()
print(end - start)

