from momaapi import MOMA
from torchvision.transforms import *
from pytorchvideo.transforms import *
from pytorchvideo.data import LabeledVideoDataset
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from pytorchvideo.data.clip_sampling import make_clip_sampler
from torch.utils.data.sampler import RandomSampler
from collections import Counter

dir_moma = '../../data/moma'


def get_act_dataset():

    moma = MOMA(dir_moma)
    ids_act_val = moma.get_ids_act(split='val')
    print("ACT IDS:", len(ids_act_val))
    paths_act_val = moma.get_paths(ids_act=ids_act_val)
    print("Num videos:", len(paths_act_val))
    anns_act_val = moma.get_anns_act(ids_act_val)
    cids_act_val = [ann_act_val.cid for ann_act_val in anns_act_val]
    occurrences = Counter(cids_act_val)
    print("Classifying on the SET:", set(cids_act_val))
    print("Occurrences:", occurrences)
    labeled_video_paths_val = [(path, {'label': cid}) for path, cid in zip(paths_act_val, cids_act_val)]
    
    NUM_SECONDS = 3
    use_ddp = False
    
    transform_val = Compose([
        ApplyTransformToKey(
            key='video',
            transform=Compose([
                UniformTemporalSubsample(30*NUM_SECONDS), 
                Lambda(lambda x: x/255.0), # Only normalization for VideoCLIP is / 255.0 
                ShortSideScale(size=256),
                CenterCrop(224)
            ])
        ),
    ])

    dataset_val = LabeledVideoDataset(
        labeled_video_paths = labeled_video_paths_val, 
        clip_sampler = ConstantClipsPerVideoSampler(clip_duration=2*32/30, clips_per_video = 1, augs_per_clip=1),
        video_sampler = DistributedSampler if use_ddp else RandomSampler,
        transform = transform_val,
        decode_audio = False
    )
    
    return dataset_val

def get_sact_dataset():

    moma = MOMA(dir_moma)
    ids_sact_val = moma.get_ids_sact(split='val')
    print("SACT IDS:", len(ids_sact_val))
    paths_sact_val = moma.get_paths(ids_sact=ids_sact_val)
    print("Num videos:", len(paths_sact_val))
    anns_sact_val = moma.get_anns_sact(ids_sact_val)
    cids_sact_val = [ann_sact_val.cid for ann_sact_val in anns_sact_val]
    occurrences = Counter(cids_sact_val)
    print("Classifying on the SET:", set(cids_sact_val))
    print("Occurrences:", occurrences)
    labeled_video_paths_val = [(path, {'label': cid}) for path, cid in zip(paths_sact_val, cids_sact_val)]
    
    NUM_SECONDS = 3
    use_ddp = False
    
    transform_val = Compose([
        ApplyTransformToKey(
            key='video',
            transform=Compose([
                UniformTemporalSubsample(30*NUM_SECONDS), 
                Lambda(lambda x: x/255.0), # Only normalization for VideoCLIP is / 255.0 
                ShortSideScale(size=256),
                CenterCrop(224)
            ])
        ),
    ])

    dataset_val = LabeledVideoDataset(
        labeled_video_paths = labeled_video_paths_val, 
        clip_sampler = ConstantClipsPerVideoSampler(clip_duration=2*32/30, clips_per_video = 1, augs_per_clip=1),
        video_sampler = DistributedSampler if use_ddp else RandomSampler,
        transform = transform_val,
        decode_audio = False
    )
    
    return dataset_val

def val_dataloader(act_type):
    """
    Create the Kinetics validation partition from the list of video labels
    in {self._DATA_PATH}/val
    """
    if act_type == 'activity':
        val_dataset = get_act_dataset()
    elif act_type == 'sub-activity':
        val_dataset = get_sact_dataset()
    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=1,
    )