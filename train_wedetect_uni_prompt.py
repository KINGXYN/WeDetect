#!/usr/bin/env python3
# Copyright (c) Tencent Inc. All rights reserved.
import argparse
import os

from mmengine.config import Config
from mmengine.runner import Runner

from mmdet.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train WeDetect-Uni prompt embeddings")
    parser.add_argument(
        "--train-ann-file",
        default="/root/userfolder/data/TCT_JPEGImages/train30000-cat10.json",
        help="path to training COCO annotation json")
    parser.add_argument(
        "--train-img-prefix",
        default="/root/userfolder/data/TCT_JPEGImages/train30000",
        help="path to training image folder")
    parser.add_argument(
        "--val-ann-file",
        default="/root/userfolder/data/TCT_JPEGImages/val10000-cat10.json",
        help="path to validation COCO annotation json")
    parser.add_argument(
        "--val-img-prefix",
        default="/root/userfolder/data/TCT_JPEGImages/val",
        help="path to validation image folder")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="number of categories in the dataset")
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=20,
        help="total training epochs")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="train batch size per GPU")
    parser.add_argument(
        "--base-lr",
        type=float,
        default=1e-3,
        help="base learning rate")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
        help="weight decay")
    parser.add_argument(
        "--work-dir",
        default="./work_dirs/wedetect_uni_prompt_tct",
        help="the directory to save logs and checkpoints")
    parser.add_argument(
        "--load-from",
        help="checkpoint file to load for initialization")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume training from the latest checkpoint")
    parser.add_argument(
        "--train-prompt-only",
        action="store_true",
        help="freeze all parameters except prompt embeddings")
    return parser.parse_args()


def freeze_except_prompt(model):
    for name, param in model.named_parameters():
        if name.startswith("embeddings"):
            param.requires_grad = True
        else:
            param.requires_grad = False


def build_config(args):
    num_classes = args.num_classes
    img_scale = (640, 640)
    text_channels = 768
    return Config(
        dict(
            custom_imports=dict(imports=["wedetect"], allow_failed_imports=False),
            find_unused_parameters=True,
            model_test_cfg=dict(
                multi_label=True,
                nms_pre=30000,
                score_thr=0.001,
                nms=dict(type="nms", iou_threshold=0.7),
                max_per_img=300,
            ),
            model=dict(
                type="SimpleYOLOWorldDetector",
                mm_neck=False,
                num_train_classes=num_classes,
                num_test_classes=num_classes,
                prompt_dim=text_channels,
                num_prompts=num_classes,
                freeze_prompt=False,
                data_preprocessor=dict(
                    type="YOLOWDetDataPreprocessor",
                    mean=[0.0, 0.0, 0.0],
                    std=[255.0, 255.0, 255.0],
                    bgr_to_rgb=True,
                ),
                backbone=dict(
                    type="MultiModalYOLOBackbone",
                    image_model=dict(
                        type="ConvNextVisionBackbone",
                        model_name="base",
                        frozen_modules=[],
                    ),
                    text_model=dict(
                        type="XLMRobertaLanguageBackbone",
                        model_name="./xlm-roberta-base/",
                        model_size="base",
                        frozen_modules=[],
                    ),
                    with_text_model=False,
                ),
                neck=dict(
                    type="CSPRepBiFPANNeck",
                    scale_factor=1.0,
                    model_size="base",
                ),
                bbox_head=dict(
                    type="YOLOWorldHead",
                    head_module=dict(
                        type="YOLOWorldHeadModule",
                        use_bn_head=True,
                        embed_dims=text_channels,
                        num_classes=num_classes,
                        model_size="base",
                        in_channels=[256, 512, 1024],
                        freeze_all=True,
                    ),
                    prior_generator=dict(
                        type="MlvlPointGenerator", offset=0.5, strides=[8, 16, 32]
                    ),
                    bbox_coder=dict(type="WeDetectDistancePointBBoxCoder"),
                    loss_cls=dict(
                        type="CrossEntropyLoss",
                        use_sigmoid=True,
                        reduction="none",
                        loss_weight=0.5,
                    ),
                    loss_bbox=dict(
                        type="mmyoloIoULoss",
                        iou_mode="ciou",
                        bbox_format="xyxy",
                        reduction="sum",
                        loss_weight=7.5,
                        return_iou=False,
                    ),
                    loss_dfl=dict(
                        type="DistributionFocalLoss",
                        reduction="mean",
                        loss_weight=1.5 / 4,
                    ),
                ),
                train_cfg=dict(
                    assigner=dict(
                        type="BatchTaskAlignedAssigner",
                        num_classes=num_classes,
                        use_ciou=True,
                        topk=10,
                        alpha=0.5,
                        beta=6.0,
                        eps=1e-9,
                    )
                ),
                test_cfg=dict(
                    multi_label=True,
                    nms_pre=30000,
                    score_thr=0.001,
                    nms=dict(type="nms", iou_threshold=0.7),
                    max_per_img=300,
                ),
            ),
            train_pipeline=[
                dict(type="LoadImageFromFile", backend_args=None),
                dict(type="LoadAnnotations", with_bbox=True, _scope_="mmdet"),
                dict(type="WeDetectKeepRatioResize", scale=img_scale),
                dict(
                    type="WeDetectLetterResize",
                    scale=img_scale,
                    allow_scale_up=False,
                    pad_val=dict(img=114),
                ),
                dict(
                    type="PackDetInputs",
                    meta_keys=(
                        "img_id",
                        "img_path",
                        "ori_shape",
                        "img_shape",
                        "scale_factor",
                        "pad_param",
                    ),
                ),
            ],
            test_pipeline=[
                dict(type="LoadImageFromFile", backend_args=None),
                dict(type="WeDetectKeepRatioResize", scale=img_scale),
                dict(
                    type="WeDetectLetterResize",
                    scale=img_scale,
                    allow_scale_up=False,
                    pad_val=dict(img=114),
                ),
                dict(type="LoadAnnotations", with_bbox=True, _scope_="mmdet"),
                dict(
                    type="PackDetInputs",
                    meta_keys=(
                        "img_id",
                        "img_path",
                        "ori_shape",
                        "img_shape",
                        "scale_factor",
                        "pad_param",
                    ),
                ),
            ],
            train_dataloader=dict(
                batch_size=args.batch_size,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                drop_last=True,
                sampler=dict(type="DefaultSampler", shuffle=True),
                dataset=dict(
                    type="WeCocoDataset",
                    data_root="",
                    ann_file=args.train_ann_file,
                    data_prefix=dict(img=args.train_img_prefix),
                    pipeline=None,
                ),
            ),
            val_dataloader=dict(
                batch_size=1,
                num_workers=2,
                persistent_workers=True,
                pin_memory=True,
                drop_last=False,
                sampler=dict(type="DefaultSampler", shuffle=False),
                dataset=dict(
                    type="WeCocoDataset",
                    data_root="",
                    ann_file=args.val_ann_file,
                    data_prefix=dict(img=args.val_img_prefix),
                    pipeline=None,
                    test_mode=True,
                ),
            ),
            test_dataloader=None,
            val_evaluator=dict(
                type="CocoMetric",
                ann_file=args.val_ann_file,
                metric="bbox",
            ),
            test_evaluator=None,
            train_cfg=dict(type="EpochBasedTrainLoop",
                           max_epochs=args.max_epochs,
                           val_interval=1),
            val_cfg=dict(type="ValLoop"),
            test_cfg=dict(type="TestLoop"),
            optim_wrapper=dict(
                type="OptimWrapper",
                optimizer=dict(
                    type="AdamW",
                    lr=args.base_lr,
                    weight_decay=args.weight_decay,
                ),
                clip_grad=dict(max_norm=10.0, norm_type=2),
            ),
            param_scheduler=[
                dict(
                    type="LinearLR",
                    start_factor=0.1,
                    by_epoch=False,
                    begin=0,
                    end=500,
                ),
                dict(
                    type="CosineAnnealingLR",
                    T_max=args.max_epochs,
                    by_epoch=True,
                    begin=0,
                    end=args.max_epochs,
                ),
            ],
            default_hooks=dict(
                checkpoint=dict(type="CheckpointHook", interval=1)
            ),
            work_dir=args.work_dir,
            load_from=args.load_from,
            resume=args.resume,
        ))


def main():
    args = parse_args()
    register_all_modules()

    cfg = build_config(args)
    cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
    cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline
    cfg.test_dataloader = cfg.val_dataloader
    cfg.test_evaluator = cfg.val_evaluator

    runner = Runner.from_cfg(cfg)
    if args.train_prompt_only:
        freeze_except_prompt(runner.model)

    os.makedirs(cfg.work_dir, exist_ok=True)
    runner.train()


if __name__ == "__main__":
    main()
