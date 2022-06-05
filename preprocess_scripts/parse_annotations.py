import os 
import scipy 
import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy.io as spio
import pickle
from collections import Counter

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def parse_emotion_data(annot_disc_array,top_k=3):
    disc_cat=[]
    #print(annot_disc_array)
    if(annot_disc_array.__class__.__name__ is 'mat_struct'):
        cat=annot_disc_array.categories
        if(isinstance(cat, str)):
            disc_cat=[cat]
    else:
        for i in np.arange(len(annot_disc_array)):
            if(isinstance(annot_disc_array[i].categories,str)):
                  disc_cat=[annot_disc_array[i].categories]
            else:
                disc_cat=disc_cat+list(annot_disc_array[i].categories)
                
    disc_cat_counter=dict(Counter(disc_cat).most_common(top_k))
    label_top_k=list(disc_cat_counter.keys())
    
    return(label_top_k)

def parse_train_data(data):
    dict_tot={}
    for i in np.arange(len(data)):
        filename=data[i].filename
        folder=data[i].folder
        img_s=data[i].image_size
        n_col=img_s.n_col
        n_row=img_s.n_row
        #db=train_data[i].original_database
        person_data=data[i].person
        # print(type(person_data))
        # print(person_data.__class__.__name__)
        dict_person={}
        if(person_data.__class__.__name__ is 'mat_struct'):
            dict_temp={}
            bbox=person_data.body_bbox
            annotation_categories=person_data.annotations_categories.categories
            valence_categories=person_data.annotations_continuous.valence
            arousal_categories=person_data.annotations_continuous.arousal
            dominance_categories=person_data.annotations_continuous.dominance
            AVD={'arousal':arousal_categories,'valence':valence_categories,'dominance':dominance_categories}
            age=person_data.age
            gender=person_data.gender
            dict_temp['bbox']=bbox
            dict_temp['disc_emotion']=annotation_categories
            dict_temp['cont_emotion']=AVD
            dict_temp['gender']=gender
            dict_temp['age']=age

            dict_person={'person_0':dict_temp,'filename':filename,'folder':folder}
        else:

            for i in np.arange(len(person_data)):
                dict_temp={}
                #     key='person_'+str(j)
                bbox=person_data[i].body_bbox
                annotation_categories=person_data[i].annotations_categories.categories
                valence_categories=person_data[i].annotations_continuous.valence
                arousal_categories=person_data[i].annotations_continuous.arousal
                dominance_categories=person_data[i].annotations_continuous.dominance
                AVD={'arousal':arousal_categories,'valence':valence_categories,'dominance':dominance_categories}
                age=person_data[i].age
                gender=person_data[i].gender

                dict_temp['bbox']=bbox
                dict_temp['disc_emotion']=annotation_categories
                dict_temp['cont_emotion']=AVD
                dict_temp['gender']=gender
                dict_temp['age']=age

                dict_person['person_'+str(i)]=dict_temp

            dict_person['filename']=filename
            dict_person['folder']=folder 
        
        dict_tot[filename]=dict_person
    return(dict_tot)


def parse_val_data(data):
    dict_tot={}
    for i in np.arange(len(data)):
        
        filename=data[i].filename
        folder=data[i].folder
        img_s=data[i].image_size
        n_col=img_s.n_col
        n_row=img_s.n_row
        
        person_data=data[i].person
        dict_person={}

        if(person_data.__class__.__name__ is 'mat_struct'):
            dict_temp={}
            # person_data_set=_todict(person_data)
            # print(person_data_set)
            bbox=person_data.body_bbox
            annotation_categories=person_data.combined_categories
            valence_categories=person_data.combined_continuous.valence
            arousal_categories=person_data.combined_continuous.arousal
            dominance_categories=person_data.combined_continuous.dominance
            
            AVD={'arousal':arousal_categories,'valence':valence_categories,'dominance':dominance_categories}
            age=person_data.age
            gender=person_data.gender
            dict_temp['bbox']=bbox
            dict_temp['disc_emotion']=annotation_categories
            dict_temp['cont_emotion']=AVD
            dict_temp['gender']=gender
            dict_temp['age']=age
            dict_person={'person_0':dict_temp,'filename':filename,'folder':folder}

        else:
            for i in np.arange(len(person_data)):
                    dict_temp={}
                    bbox=person_data[i].body_bbox
                    #use combined categories and continuous annotations here 
                    #for discrete case we consider the distribution of emotions annotated for a single sample by multiple annotators

                    annotation_categories=person_data[i].combined_categories #dim mismatch here
                    annot_disc_array=person_data[i].annotations_categories
                    label_top_k=parse_emotion_data(annot_disc_array)
                    
                    #for arousal valence and dominance we consider the average ratings 
                    valence_categories=person_data[i].combined_continuous.valence
                    arousal_categories=person_data[i].combined_continuous.arousal
                    dominance_categories=person_data[i].combined_continuous.dominance
                    AVD={'arousal':arousal_categories,'valence':valence_categories,'dominance':dominance_categories}
                    age=person_data[i].age
                    gender=person_data[i].gender
                    dict_temp['bbox']=bbox
                    dict_temp['disc_emotion']=annotation_categories
                    dict_temp['disc_emotion_top_k']=label_top_k
                    dict_temp['cont_emotion']=AVD
                    dict_temp['gender']=gender
                    dict_temp['age']=age
                    dict_person['person_'+str(i)]=dict_temp

        dict_person['filename']=filename
        dict_person['folder']=folder 
        dict_tot[filename]=dict_person

    return(dict_tot)


annotation_file="/bigdata/digbose92/Emotic/Annotations/Annotations/Annotations.mat"
annotation_mat = loadmat(annotation_file,struct_as_record=False,squeeze_me=True)
print(annotation_mat.keys())
#data_dict=_check_keys(annotation_mat)
#print(data_dict)
train_data=annotation_mat['train']
val_data=annotation_mat['val']
test_data=annotation_mat['test']

dict_train=parse_train_data(train_data)
dict_val=parse_val_data(val_data)
dict_test=parse_val_data(test_data)

with open("/bigdata/digbose92/Emotic/pkl_files/train_data.pkl", "wb") as f:
     pickle.dump(dict_train, f)

with open("/bigdata/digbose92/Emotic/pkl_files/val_data.pkl", "wb") as f:
     pickle.dump(dict_val, f)

with open("/bigdata/digbose92/Emotic/pkl_files/test_data.pkl", "wb") as f:
     pickle.dump(dict_test, f)

print(len(dict_train),len(dict_val),len(dict_test))
#dict structure is simple with first key as filename with subkeys as folder, 
# person_1: bbox, disc_emotion, cont_emotion, gender, age 
# person_2: bbox, disc_emotion, cont_emotion, gender, age 
# dict_tot={}
# for i in np.arange(len(train_data)):

#     filename=train_data[i].filename
#     folder=train_data[i].folder

#     img_s=train_data[i].image_size
#     n_col=img_s.n_col
#     n_row=img_s.n_row

#     #db=train_data[i].original_database
#     person_data=train_data[i].person
#     # print(type(person_data))
#     # print(person_data.__class__.__name__)
#     dict_person={}
#     if(person_data.__class__.__name__ is 'mat_struct'):
#         dict_temp={}
#         bbox=person_data.body_bbox
#         annotation_categories=person_data.annotations_categories.categories
#         valence_categories=person_data.annotations_continuous.valence
#         arousal_categories=person_data.annotations_continuous.arousal
#         dominance_categories=person_data.annotations_continuous.dominance
#         AVD={'arousal':arousal_categories,'valence':valence_categories,'dominance':dominance_categories}
#         age=person_data.age
#         gender=person_data.gender

#         dict_temp['bbox']=bbox
#         dict_temp['disc_emotion']=annotation_categories
#         dict_temp['cont_emotion']=AVD
#         dict_temp['gender']=gender
#         dict_temp['age']=age

#         dict_person={'person_0':dict_temp,'filename':filename,'folder':folder}
#     else:

#         for i in np.arange(len(person_data)):
#             dict_temp={}
#             #     key='person_'+str(j)
#             bbox=person_data[i].body_bbox
#             annotation_categories=person_data[i].annotations_categories.categories
#             valence_categories=person_data[i].annotations_continuous.valence
#             arousal_categories=person_data[i].annotations_continuous.arousal
#             dominance_categories=person_data[i].annotations_continuous.dominance
#             AVD={'arousal':arousal_categories,'valence':valence_categories,'dominance':dominance_categories}
#             age=person_data[i].age
#             gender=person_data[i].gender

#             dict_temp['bbox']=bbox
#             dict_temp['disc_emotion']=annotation_categories
#             dict_temp['cont_emotion']=AVD
#             dict_temp['gender']=gender
#             dict_temp['age']=age

#             dict_person['person_'+str(i)]=dict_temp

#         dict_person['filename']=filename
#         dict_person['folder']=folder 
    
#     dict_tot[filename]=dict_person

# with open("/bigdata/digbose92/Emotic/pkl_files/train_data.pkl", "wb") as f:
#     pickle.dump(dict_tot, f)


# val_data=annotation_mat['val']

# dict_tot={}
# for i in np.arange(len(val_data)):
#     #print(val_data[i])
#     filename=val_data[i].filename
    
#     folder=val_data[i].folder
#     print(filename,folder)
#     img_s=val_data[i].image_size
#     n_col=img_s.n_col
#     n_row=img_s.n_row

#     #db=train_data[i].original_database
#     person_data=val_data[i].person
#     print(type(person_data))
#     #print(len(person_data))
#     #print(person_data)
#     # print(type(person_data))
#     # print(person_data.__class__.__name__)
#     dict_person={}
    
#     for i in np.arange(len(person_data)):
#             dict_temp={}
#             #     key='person_'+str(j)
#             bbox=person_data[i].body_bbox
#             #use combined categories and continuous annotations here 

#             annotation_categories=person_data[i].combined_categories#dim mismatch here
#             print(annotation_categories)
#             valence_categories=person_data[i].combined_continuous.valence
#             arousal_categories=person_data[i].combined_continuous.arousal
#             dominance_categories=person_data[i].combined_continuous.dominance


#             AVD={'arousal':arousal_categories,'valence':valence_categories,'dominance':dominance_categories}
#             age=person_data[i].age
#             gender=person_data[i].gender
#             dict_temp['bbox']=bbox
#             dict_temp['disc_emotion']=annotation_categories
#             dict_temp['cont_emotion']=AVD
#             dict_temp['gender']=gender
#             dict_temp['age']=age
#             dict_person['person_'+str(i)]=dict_temp

#     dict_person['filename']=filename
#     dict_person['folder']=folder 
#     dict_tot[filename]=dict_person

# # with open("/bigdata/digbose92/Emotic/pkl_files/val_data.pkl", "wb") as f:
# #     pickle.dump(dict_tot, f)

# # with open("/bigdata/digbose92/Emotic/pkl_files/train_data.pkl", "wb") as f:
# #     pickle.dump(dict_tot, f)
# #print(dict_temp)
# #print(img_data)
