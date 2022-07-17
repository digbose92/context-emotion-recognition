from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
from PIL import Image
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)
import torch.nn as nn
from tqdm import tqdm 
import numpy as np 
import pandas as pd
import os 
import pickle

activation = {}
def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activation[name] = output.detach()
  return hook

##### load the model #####
model_file="/data/digbose92/codes/Emotic_experiments/vit_scene_model/20220416-183337_vit_pretrained_places_365_model/20220416-183337_vit_base_model_best_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_total=torch.load(model_file)
model_state_dict=model_total.module.state_dict()

#declare vit model here 
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k") #doesn't matter if its pretrained :)
num_classes=365

id2label_file='/data/digbose92/codes/Emotic_experiments/vit_scene_mapping/places365_id2label.pkl'
with open(id2label_file,'rb') as f:
        id2label=pickle.load(f)

label2id_file='/data/digbose92/codes/Emotic_experiments/vit_scene_mapping/places365_label2id.pkl'
with open(label2id_file,'rb') as f:
        label2id=pickle.load(f)

model=ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                num_labels=num_classes,
                                id2label=id2label,
                                label2id=label2id)

model.load_state_dict(model_state_dict)
model=model.eval()
model=model.to(device)

# #normalization
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k") #doesn't matter if its pretrained :)
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
_val_transforms = Compose(
        [
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize,
        ]
    )

#test feature value on one of the images
#image_file="/bigdata/digbose92/Emotic/ade20k/images/labelme_vjsynrlwrnjrcbw.jpg"

csv_file="/bigdata/digbose92/Emotic/captions/caption_results_framesdb_emodb_small_mscoco_ade20k.csv"
csv_data=pd.read_csv(csv_file)
image_filename_list=list(csv_data['image_id'])
dest_folder="/bigdata/digbose92/Emotic/vit_scene_features/"

feat_dict=dict()
#cnt=0
for image_file in tqdm(image_filename_list):

    img_key=image_file.split("/")[-1].split(".")[0]
    img=Image.open(image_file).convert('RGB')
    img=_val_transforms(img)
    img=img.unsqueeze(0)
    img=img.to(device)
    

    h1=model.vit.layernorm.register_forward_hook(getActivation('feats'))
    #print(activation)
    with torch.no_grad():
        val=model(img)
        layer_norm_data=activation['feats'].squeeze(0).cpu().numpy() #[1,197,768]

    feat_dict[img_key]=layer_norm_data
    
    
with open(os.path.join(dest_folder,"vit_scene_feature_layer_norm.pkl"),"wb") as f:
        pickle.dump(feat_dict,f)




    #     log_softmax=nn.LogSoftmax(dim=-1)
    #     values=log_softmax(val.logits).to('cpu')
    #     y_pred=torch.max(values, 1)[1].numpy()[0]

    # with open('/data/digbose92/codes/Emotic_experiments/vit-experiments/scripts/places365.txt', 'r') as f:
    #     labels = [line.rstrip('\n') for line in f]

    # label_dict=dict()
    # for l in labels:
    #     label_list=l.split(" ")
    #     label_dict[label_list[1]]=label_list[0]

    # print('Predicted label:%s' %(label_dict[str(y_pred)]))



    # # for name, m in model.named_modules():
    # #     print(name, m)
    #     #print(label_list)
        
    #     #print(labels[0])
    #     # label_list=labels.split(" ")
    #     # print(label_list)
    # # #print(y_pred)
    # # print(model.module)
    # # model.vit.layernorm.register_forward_hook(getActivation('feats'))
    # # print(activation)



