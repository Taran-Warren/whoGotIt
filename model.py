import warnings
warnings.filterwarnings("ignore")
import torch
import open_clip
from PIL import Image
from mobileclip.modules.common.mobileone import reparameterize_model

model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP2-S0', pretrained=r'C:\Users\Warren\.cache\huggingface\hub\models--apple--MobileCLIP2-S2\snapshots\72424e7025436db18f15c3eff6ee8c7c15ad4481\mobileclip2_s2.pt')
#model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP2-S0', pretrained=r'C:\Users\Warren\.cache\huggingface\hub\models--apple--MobileCLIP2-S0\snapshots\3136ea51c8ed56b9f9abfab04cb816735aaad6cb\mobileclip2_s0.pt')
model = reparameterize_model(model)
model.eval()
image1 = preprocess(Image.open("test_1a.jpg")).unsqueeze(0)
image2 = preprocess(Image.open("test_1b.jpg")).unsqueeze(0)
image3 = preprocess(Image.open("test_2a.jpg")).unsqueeze(0)
image4 = preprocess(Image.open("test_3a.jpg")).unsqueeze(0)

with torch.no_grad():
    embedding1 = model.encode_image(image1)
    embedding2 = model.encode_image(image2)
    embedding3 = model.encode_image(image3)
    embedding4 = model.encode_image(image4)

similarity_same = torch.nn.functional.cosine_similarity(embedding1, embedding2)
similarity_diff = torch.nn.functional.cosine_similarity(embedding1, embedding4)
print("Same object similarity:", similarity_same.item())
print("Different object similarity:", similarity_diff.item())