_base_ = './dino_config.py'

# model settings
model = dict(
    name_path='evaluation/ClearCLIP/configs/cls_voc21.txt',
    prob_thd=0.5
)

# dataset settings
dataset_type = 'PascalVOCDataset'
data_root = '/network/scratch/q/qian.yang/segmentation_datasets/pascalvoc20/VOCdevkit/VOC2012'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 448), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))