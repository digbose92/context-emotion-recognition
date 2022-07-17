#text+scene information must be combined through MCAN 
#person information must be processed by adapter layers
import torch
import torch.nn as nn 
import torchvision
from MCAN_model import *

class Adapter_model(nn.Module):
    def __init__(self,in_dim,mid_dim):

        self.in_dim=in_dim 
        self.mid_dim=mid_dim 

        #upsampled and downsampled operations
        self.fc_downsample=nn.Linear(self.in_dim,self.mid_dim) #downsamples the input data to mid_dim dimension
        self.fc_upsample=nn.Linear(self.mid_dim,self.in_dim) #upsamples the downsampled data to in_dim dimension
        self.GELU_op=nn.GELU()

    def forward(self,x):

        downsample_feat=self.fc_downsample(x)
        downsample_feat=self.GELU_op(downsample_feat)
        upsample_feat=self.fc_upsample(downsample_feat)
        x=x+upsample_feat
        return(x)


class text_scene_comb_person_model(nn.Module):

    def __init__(self,mcan_config,
                    in_feat_dim=2048,
                    mid_feat_dim=512,
                    scene_feat_dim=768,
                    num_adapter_layers=2,
                    fusion_option='concat',
                    num_classes=26
                    ):

        self.in_feat_dim=in_feat_dim
        self.mid_feat_dim=mid_feat_dim
        self.scene_feat_dim=scene_feat_dim
        self.num_adapter_layers=num_adapter_layers
        self.mcan_config=mcan_config
        self.fusion_option=fusion_option
        self.num_classes=num_classes 
        #mcan_config is a class with items LAYER, HIDDEN_SIZE, MULTI_HEAD, DROPOUT_R

        self.person_adapter_module=nn.ModuleList([Adapter_model(self.in_feat_dim,self.mid_feat_dim) for _ in range(self.num_adapter_layers)])

        #person and scene information fusion model
        self.person_scene_model=MCA_ED(self.mcan_config)

        #classifier layer
        self.inp_cls_feat_dim=(2*self.scene_feat_dim+in_feat_dim)
        self.classifier_fc=nn.Linear(self.inp_cls_feat_dim,self.num_classes)

    def forward(self,person_feat,scene_feat,text_feat,text_mask):

        person_feat=self.person_adapter_module(person_feat)

        #coattention modeling by MCAN architecture
        mca_feat=self.person_scene_model(scene_feat,text_feat,None,text_mask)

        if(self.fusion_option=='concat'):
            ov_feat=torch.cat([person_feat,mca_feat],dim=1)

        logits=self.classifier(ov_feat)

        return(logits)














