from momaapi import MOMA
import torch
import sys
from mmpt.models import MMPTModel, MMPTClassifier
from torchvision.io import read_video, write_video
from torch.nn import Sequential
import torchvision.transforms as Transforms
import skvideo.io  
import numpy as np
import yaml

import pytorch_lightning
from preprocessing import val_dataloader

def predict(moma, moma_acts, act_type, act_names):
    
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
    dataloader = val_dataloader(act_type)
    
    
    # Init model
    classifier = MMPTClassifier.from_pretrained("projects/retri/videoclip/how2.yaml")
    classifier.set_class_names(act_names)    
    
    nb_classes = len(moma_acts)
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    top5_preds = torch.zeros(nb_classes) # Num times the class was chosen in the top 5 (when it was the correct)
    num_occurences = torch.zeros(nb_classes) # Num times the class was the correct answer
    y_preds = []
    y_labels = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs = data['video'].cuda()
            inputs = inputs.permute(0, 2, 3, 4, 1)
            inputs = inputs.view(1, 3, 30, 224, 224, 3)
            classes = data['label']
            class_label = classes.item()
            num_occurences[classes.item()] += 1

            outputs = classifier(inputs)
            
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
                print("pred:", moma_acts[pred], "actual:", moma_acts[actual], pred, actual)

    print(confusion_matrix)
    print(confusion_matrix.diag()/confusion_matrix.sum(1))
    print(top5_preds / num_occurences)
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

    
    
        
def main():
    

    if len(sys.argv) == 1:
        print("ERROR: inference.py has required argument [config file]")
        return
    
    

    with open(sys.argv[1]) as file:
        data = yaml.full_load(file)
        print(data)
    
    act_type = data['activity_type']

    moma_acts = data['class_names'] 
    act_names = None
    if 'renamed_classes' in data:
        act_names = data['renamed_classes']
    
    dir_moma = '../../data/moma'
    moma = MOMA(dir_moma)
    
    predict(moma, moma_acts, act_type, act_names)
    
    
    

# Takes argument [YAML config file] (see configs/activity.yaml
if __name__ == '__main__':
    main()
