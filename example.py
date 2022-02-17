import torch

from mmpt.models import MMPTModel
from torchvision.io import read_video, write_video
from torch.nn import Sequential
import torchvision.transforms as Transforms
from scripts.video_feature_extractor.preprocessing import Preprocessing
import skvideo.io  
import numpy as np

#video_frames, _, info = read_video('tennis-game.mkv', start_pts #= 60.0, end_pts= 62.38, pts_unit='sec') # T, H, W, C
#T, H, W, C = video_frames.size()
#print(video_frames.size())

#fps = int(info["video_fps"])
#print("FPS:", fps)
#print(video_frames.size())

#video_frames = video_frames.view((1, -1, 30, H, W, C))

videodata = skvideo.io.vread("tennis-game.mkv")
videodata = videodata[1000:1060,:,:,:]
_, H, W, C = videodata.shape
videodata = np.reshape(videodata, (1, 2, 30, H, W, C))
print(videodata.shape)

model, tokenizer, aligner = MMPTModel.from_pretrained(
    "projects/retri/videoclip/how2.yaml")

model.eval()

#transforms = Sequential(
#    Transforms.CenterCrop(224), 
#    Transforms.ConvertImageDtype(torch.float))
    #T.Normalize([mean_R, mean_G, mean_B], [std_R, std_G, std_B],
    
#frames = transforms(video_frames)
preprocessor = Preprocessing('s3d')

# B, T, FPS, H, W, C (VideoCLIP is trained on 30 fps of s3d)
text_to_try = ["cats and dogs", "ordering a meal", "two men", "men playing sports", "tennis", "men playing tennis"]

video_frames = torch.from_numpy(videodata / 255.0).float() 
for text in text_to_try:                          
    caps, cmasks = aligner._build_text_seq(
        tokenizer(text, add_special_tokens=False)["input_ids"]
    )

    caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1

    with torch.no_grad():
        output = model(video_frames, caps, cmasks, return_score=True)
    print(text, output["score"])  # dot-product
