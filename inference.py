from momaapi import MOMA
import torch
import sys
from mmpt.models import MMPTModel, MMPTClassifier
from torchvision.io import read_video, write_video
from torch.nn import Sequential
import torchvision.transforms as Transforms
import numpy as np
import yaml

import pytorch_lightning
from preprocessing import val_dataloader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pdb

def predict(moma, moma_acts, act_type, act_names, num_seconds):
    
    is_renamed = True
    
    if type(moma_acts) == str: 
        if moma_acts == "all":
            if act_type == 'sub-activity':
                print("MOMA TAXONOMY:", moma_acts)
                moma_acts = moma.taxonomy['sact']
            elif act_type == 'activity':
                moma_acts = moma.taxonomy['act']
                print("MOMA TAXONOMY:", moma_acts)

            else:
                print("ERROR: Not supported activity type")
        else:
            print("Only support string type == 'all'")
    
    if act_names == None:
        is_renamed = False
        act_names = moma_acts
            
    # Init dataset
    dataloader, total_len = val_dataloader(act_type, moma, num_seconds)
    act_names_fn = '/home/durante/low-shot/class_names/'
    if act_type == 'activity':
        act_names_fn += 'act_class_names.npy'
    elif act_type == 'sub-activity':
        act_names_fn += 'sact_class_names.npy'
    act_names = np.load(act_names_fn)
    
    
    # Init model
    classifier = MMPTClassifier.from_pretrained("projects/retri/videoclip/how2.yaml", embed_extractor=True)
    classifier.set_class_names(act_names)
    
    # Extract video class embeddings (and act names)
    text_embeds = classifier.text_embeds.detach().cpu().numpy()
    text_embed_fn = "/home/durante/low-shot/video_clip/" 
    if act_type == "activity":
        text_embed_fn += 'act/text_embeds.npy'
    elif act_type == 'sub-activity':
        text_embed_fn += 'sact/text_embeds.npy'
        
    np.save(text_embed_fn, text_embeds)

    
    
    nb_classes = len(moma_acts)
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    top5_preds = torch.zeros(nb_classes) # Num times the class was chosen in the top 5 (when it was the correct)
    num_occurences = torch.zeros(nb_classes) # Num times the class was the correct answer
    y_preds = []
    y_labels = []
    # acts are 226
    # sacts
    video_embeds = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, total=total_len)):
            #print(i)
            inputs = data['video'].cuda()
            inputs = inputs.permute(0, 2, 3, 4, 1)
            inputs = inputs.view(1, num_seconds, 30, 224, 224, 3)
            classes = data['label']
            class_label = classes.item()
            num_occurences[classes.item()] += 1

            outputs = classifier(inputs)
            video_embeds.append(np.squeeze(outputs.cpu().numpy()))
            y_labels.append(class_label)
            """
            y_labels.append(class_label)            
            _, preds = torch.max(outputs, 1)
            y_preds.append(preds[0].item())
            for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
            
            topk = torch.topk(outputs, k=5)[1] # Get the indices
            if class_label in topk:
                top5_preds[class_label] += 1
            
            if i % 20 == 0:
                #torch.save(confusion_matrix, 'confusion_matrix_' + act_type + '.pt')
                pred, actual = preds.item(), classes.item()
                print("step", i)
                print("pred:", moma_acts[pred], "actual:", moma_acts[actual], pred, actual)"""
    
    video_embed_fn = "/home/durante/low-shot/video_clip/" 
    if act_type == "activity":
        video_embed_fn += 'act/vid_embeds.npy'
    elif act_type == 'sub-activity':
        video_embed_fn += 'sact/vid_embeds.npy'
    np.save(video_embed_fn, np.asarray(video_embeds))
    
    labels_fn = "/home/durante/low-shot/video_clip/" 
    if act_type == "activity":
        labels_fn += 'act/labels.npy'
    elif act_type == 'sub-activity':
        labels_fn += 'sact/labels.npy'
    np.save(labels_fn, np.asarray(y_labels))
    
    """
    print("accuracy:", accuracy_score(y_preds, y_labels))
    
    if not is_renamed:
        torch.save(top5_preds, 'top5_preds_' + act_type + '.pt')
        torch.save(num_occurences, 'num_occurences_' + act_type + '.pt')
        torch.save(confusion_matrix, 'confusion_matrix_' + act_type + '.pt')
        np.save('y_preds_' + act_type + '.npy', y_preds)
        np.save('y_labels_' + act_type + '.npy', y_labels)
    else:
        torch.save(top5_preds, 'top5_preds_renamed_' + act_type + '.pt')
        torch.save(num_occurences, 'num_occurences_renamed_' + act_type + '.pt')
        torch.save(confusion_matrix, 'confusion_matrix_renamed_' + act_type + '.pt')
        np.save('y_preds_renamed_' + act_type + '.npy', y_preds)
        np.save('y_labels_renamed_' + act_type + '.npy', y_labels)
    """
    
    
        
def main():
    

    if len(sys.argv) == 1:
        print("ERROR: inference.py has required argument [config file]")
        return
    
    

    with open(sys.argv[1]) as file:
        data = yaml.full_load(file)
        print(data)
    
    act_type = data['activity_type']

    moma_acts = data['class_names'] 
    num_seconds = 3 # Default value is 3 second clips
    if 'num_seconds' in data:
        num_seconds = data['num_seconds']
    
    act_names = None
    if 'renamed_classes' in data:
        act_names = data['renamed_classes']
    dir_moma = '../../ssd/data/moma/'
    moma = MOMA(dir_moma, load_val=True, paradigm='few-shot')
    
    predict(moma, moma_acts, act_type, act_names, num_seconds)
    
    
    

# Takes argument [YAML config file] (see configs/activity.yaml
if __name__ == '__main__':
    main()
