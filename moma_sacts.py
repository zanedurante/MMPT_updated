from momaapi import MOMAAPI
import torch

from mmpt.models import MMPTModel
from torchvision.io import read_video, write_video
from torch.nn import Sequential
import torchvision.transforms as Transforms
import skvideo.io  
import numpy as np

moma = MOMAAPI('../../data/moma')

moma_sacts = [
    "the barber is applying hair products or shaving cream",
    "the adult is feeding the child",
    "the adult is holding the child on the bike",
    "no meaning to see if length effects anything",
    "a bunch of random filler words with no meaning to see if length effects anything",
    "a bunch of random filler words",
   # "the firefighters are extinguishing fire",
]

model, tokenizer, aligner = MMPTModel.from_pretrained(
            "projects/retri/videoclip/how2.yaml")

print(moma_sacts)

for correct_idx, activity in enumerate(moma_sacts):
    print("Category:", activity)
    sact_ids = moma.get_ids_sact(cnames_sact = [activity])
    print("Number in category", len(sact_ids))
    num_examples = 10
    num_correct = 0
    paths = moma.get_paths(ids_sact=sact_ids[:num_examples])
    print("PATH LENGTH:", len(paths))
    model.eval().to('cuda')

    for path in paths:
        videodata = skvideo.io.vread(path)
        L, H, W, C = videodata.shape
        if L / 30 != 0:
            extra_frames = L % 30
            videodata = videodata[extra_frames:]
        L = len(videodata)
        if L > 240:
            # Grab middle 240 frames
            videodata = videodata[L//2-120:L//2+120] # Cap videodata to first 240 frames
        videodata = np.reshape(videodata, (1, -1, 30, H, W, C))
        # B, T, FPS, H, W, C (VideoCLIP is trained on 30 fps of s3d)
        text_to_try = moma_sacts

        video_frames = torch.from_numpy(videodata / 255.0).cuda().float() 
        scores = []
        for text in text_to_try:                          
            caps, cmasks = aligner._build_text_seq(
                tokenizer(text, add_special_tokens=False)["input_ids"]
            )

            caps, cmasks = caps[None, :].cuda(), cmasks[None, :].cuda()  # bsz=1
            with torch.no_grad():
                # Goes here first
                output = model(video_frames, caps, cmasks, return_score=True)
            #print("Text:", "'" + text + "'", "score:", output["score"].item())  # dot-product
            scores.append(output["score"].item())
        
        pred = np.argmax(scores)
        if pred == correct_idx:
            num_correct += 1
        print("Predicted class", moma_sacts[pred])
    
    print("Accuracy for class", activity, num_correct / num_examples)
        
