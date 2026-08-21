import torch
import open_clip
from PIL import Image
from mobileclip.modules.common.mobileone import reparameterize_model

model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP2-S0', pretrained=r'C:\Users\Warren\.cache\huggingface\hub\models--apple--MobileCLIP2-S0\snapshots\3136ea51c8ed56b9f9abfab04cb816735aaad6cb\mobileclip2_s0.pt')
model = reparameterize_model(model)
model.eval()