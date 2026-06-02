import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2,3,4')
import shutil
import argparse
import time
import random
from datetime import datetime
from collections import defaultdict
import numpy as np
np.set_printoptions(suppress=True)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from tensorboardX import SummaryWriter

from mvn import datasets
from mvn.models.DGPose import DepthGuidedPose
from mvn.models.DGPose_rdga_cmfr import DepthGuidedPoseRDGACMFR
from mvn.models.DGPose_posealign import DepthGuidedPosePoseAlign
from mvn.models.DGPose_oardg import DepthGuidedPoseOARDG
from mvn.models.DGPose_dlst import DepthGuidedPoseDLST
from mvn.models.DGPose_dlst_cmfm import DepthGuidedPoseDLSTCMFM
from mvn.models.DGPose_depthflow_dlst import DepthFlowPoseDLST
from mvn.models.DGPose_ude_dlst import DepthGuidedPoseUDEDLST
from mvn.models.DGPose_dlst_ude import DepthGuidedPoseDLSTUDE
from mvn.models.DGPose_rgbflow_single import RGBFlowPoseSingle
from mvn.models.DGPose_rgbflow_capf import RGBFlowPoseCAPF
from mvn.models.DGPose_rgbflow_mfce import RGBFlowPoseMFCE
from mvn.models.DGPose_rgbflow_mfce_separate import RGBFlowPoseMFCESeparate
from mvn.models.DGPose_rgbflow_jgms import RGBFlowPoseJGMS
from mvn.models.DGPose_rgbflow_cads import RGBFlowPoseCADS
from mvn.models.DGPose_rgbflow_rrms import RGBFlowPoseRRMS
from mvn.models.DGPose_rgbflow_mrrdce import RGBFlowPoseMRRDCE
from mvn.models.DGPose_rgbflow_mfas import RGBFlowPoseMFAS
from mvn.models.DGPose_rgbflow_affs import RGBFlowPoseAFFS
from mvn.models.DGPose_rgbflow_mes import RGBFlowPoseMES
from mvn.models.DGPose_rgbflow_aofs import RGBFlowPoseAOFS
from mvn.models.DGPose_rgbflow_lmrs import RGBFlowPoseLMRS
from mvn.models.DGPose_rgbflow_cmff import RGBFlowPoseCMFF
from mvn.models.DGPose_rgbflow_mces import RGBFlowPoseMCES
from mvn.models.DGPose_rgbflow_cfuafs import RGBFlowPoseCFUAFS
from mvn.models.DGPose_rgbflow_uafs import RGBFlowPoseUAFS
from mvn.models.loss import MPJPE, KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, BNNLoss, DepthLayoutLoss, DepthOrderingLoss

from mvn.utils import misc
from mvn.utils.cfg import config, update_config, update_dir
from mvn.datasets import utils as dataset_utils

print(torch.cuda.is_available())

joints_left = [4, 5, 6, 11, 12, 13] 
joints_right = [1, 2, 3, 14, 15, 16]
FLOW_MODELS = {
    "RGBFlowPoseSingle",
    "RGBFlowPoseCAPF",
    "RGBFlowPoseMFCE",
    "RGBFlowPoseMFCESeparate",
    "RGBFlowPoseJGMS",
    "RGBFlowPoseCADS",
    "RGBFlowPoseRRMS",
    "RGBFlowPoseMRRDCE",
    "RGBFlowPoseMFAS",
    "RGBFlowPoseAFFS",
    "RGBFlowPoseMES",
    "RGBFlowPoseAOFS",
    "RGBFlowPoseLMRS",
    "RGBFlowPoseCMFF",
    "RGBFlowPoseMCES",
    "RGBFlowPoseCFUAFS",
    "RGBFlowPoseUAFS",
}
FLOW_UTILITY_MODELS = {
    "RGBFlowPoseCFUAFS",
}
DEPTH_FLOW_MODELS = {
    "DepthFlowPoseDLST",
}
DLST_ORDER_MODELS = {
    "DepthGuidedPoseDLST",
    "DepthGuidedPoseDLSTCMFM",
    "DepthFlowPoseDLST",
}


def posealign_diagnostics(rel_pred, z_anchor, keypoints_3d_gt):
    if rel_pred is None or z_anchor is None:
        return {}
    if keypoints_3d_gt.dim() == 4:
        keypoints_3d_gt = keypoints_3d_gt.squeeze(1)
    gt_z = keypoints_3d_gt[..., 2]
    gt_z = gt_z - gt_z.mean(dim=1, keepdim=True)
    gt_z = gt_z / gt_z.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)

    pred_rel = rel_pred.detach()
    target_rel = torch.tanh(gt_z.unsqueeze(2) - gt_z.unsqueeze(1))
    b, j, _ = pred_rel.shape
    mask = ~torch.eye(j, device=pred_rel.device, dtype=torch.bool).unsqueeze(0).expand(b, -1, -1)
    sign_acc = (torch.sign(pred_rel[mask]) == torch.sign(target_rel[mask])).float().mean()
    rel_abs_err = (pred_rel[mask] - target_rel[mask]).abs().mean()

    pred_z = z_anchor.detach().squeeze(-1)
    pred_z = pred_z - pred_z.mean(dim=1, keepdim=True)
    gt_z_centered = gt_z - gt_z.mean(dim=1, keepdim=True)
    corr = (pred_z * gt_z_centered).mean(dim=1) / (
        pred_z.std(dim=1, unbiased=False).clamp_min(1e-6) *
        gt_z_centered.std(dim=1, unbiased=False).clamp_min(1e-6)
    )
    return {
        "posealign_rel_sign_acc": sign_acc.item(),
        "posealign_rel_abs_err": rel_abs_err.item(),
        "posealign_z_corr": corr.mean().item(),
    }


def dlst_diagnostics(rel_depth, layer_assign, keypoints_3d_gt, margin=0.05):
    if rel_depth is None or layer_assign is None:
        return {}
    if keypoints_3d_gt.dim() == 4:
        keypoints_3d_gt = keypoints_3d_gt.squeeze(1)
    gt_z = keypoints_3d_gt[..., 2]

    eval_margin = margin
    if gt_z.detach().abs().median() > 10.0 and eval_margin < 1.0:
        eval_margin = eval_margin * 1000.0
    diff = gt_z.unsqueeze(1) - gt_z.unsqueeze(2)
    target = diff.sign()

    b, j, _ = rel_depth.shape
    eye = torch.eye(j, device=rel_depth.device, dtype=torch.bool).unsqueeze(0).expand(b, -1, -1)
    mask = (diff.abs() > eval_margin) & (~eye)

    pred_sign = torch.sign(rel_depth.detach())
    if mask.any():
        sign_acc = (pred_sign[mask] == target[mask]).float().mean().item()
        resolved = (pred_sign[mask] != 0).float().mean().item()
    else:
        sign_acc = 0.0
        resolved = 0.0

    assign = layer_assign.detach().clamp_min(1e-8)
    entropy = -(assign * assign.log()).sum(dim=-1).mean()
    entropy = entropy / np.log(assign.shape[-1])
    usage = assign.mean(dim=(0, 1))

    return {
        "dlst_rel_sign_acc": sign_acc,
        "dlst_rel_resolved": resolved,
        "dlst_assign_entropy": entropy.item(),
        "dlst_layer_usage_min": usage.min().item(),
        "dlst_layer_usage_max": usage.max().item(),
    }

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path, where config file is stored")
    parser.add_argument('--eval', action='store_true', help="If set, then only evaluation will be done")
    parser.add_argument('--eval_dataset', type=str, default='val', help="Dataset split on which evaluate. Can be 'train' and 'val'")
    parser.add_argument('--checkpoint_path', type=str, default='', help="Path to checkpoint for eval (overrides default)")
    parser.add_argument('--init_checkpoint_path', type=str, default='', help="Path to checkpoint for partial initialization before training")
    parser.add_argument('--freeze_except', type=str, default='', help="Comma-separated parameter name substrings to keep trainable after init")

    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1, help="Local rank of the process on the node")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--sync_bn', action='store_true', help="If set, then utilize pytorch convert_syncbn_model")

    parser.add_argument("--logdir", type=str, default="logs/", help="Path, where logs will be stored")
    parser.add_argument("--azureroot", type=str, default="", help="Root path, where codes are stored")

    parser.add_argument("--frame", type=int, default=1, help="Frame number to be used.")
    parser.add_argument("--backbone", type=str, default='hrnet_32', choices=['hrnet_32', 'hrnet_48'], help="2D pose feature backbone.")

    args = parser.parse_args()
    # update config
    update_config(args.config)
    update_dir(args.azureroot, args.logdir)
    config.model.backbone.type = args.backbone
    return args


def setup_human36m_dataloaders(config, is_train, distributed_train, rank = None, world_size = None):
    if config.model.name in FLOW_MODELS:
        aux_image_path = config.dataset.flow_image_path
        aux_format = config.dataset.flow_format
        flow_image_path = None
        flow_format = config.dataset.flow_format
    elif config.model.name in DEPTH_FLOW_MODELS:
        aux_image_path = config.dataset.depth_image_path
        aux_format = config.dataset.depth_format
        flow_image_path = config.dataset.flow_image_path
        flow_format = config.dataset.flow_format
    else:
        aux_image_path = config.dataset.depth_image_path
        aux_format = config.dataset.depth_format
        flow_image_path = None
        flow_format = config.dataset.flow_format
    train_dataloader = None
    if is_train:
        # train
        train_dataset = eval('datasets.' + config.dataset.train_dataset)(
            root=config.dataset.root,
            pred_results_path=config.train.pred_results_path,
            depth_image_path=aux_image_path,
            flow_image_path=flow_image_path,
            train=True,
            test=False,
            image_shape=config.model.image_shape,
            labels_path=config.dataset.train_labels_path,
            with_damaged_actions=config.train.with_damaged_actions,
            scale_bbox=config.train.scale_bbox,
            kind=config.kind,
            undistort_images=config.train.undistort_images,
            ignore_cameras=config.train.ignore_cameras,
            crop=config.train.crop,
            erase=config.train.erase,
            data_format=config.dataset.data_format,
            depth_format=aux_format,
            flow_format=flow_format,
            frame=args.frame
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed_train else None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=config.train.shuffle and (train_sampler is None), # debatable
            sampler=train_sampler,
            num_workers=config.train.num_workers,
            worker_init_fn=dataset_utils.worker_init_fn,
            pin_memory=True
        )

    # val
    val_dataset = eval('datasets.' + config.dataset.val_dataset)(
        root=config.dataset.root,
        pred_results_path=config.val.pred_results_path,
        depth_image_path=aux_image_path,
        flow_image_path=flow_image_path,
        train=False,
        test=True,
        image_shape=config.model.image_shape,
        labels_path=config.dataset.val_labels_path,
        with_damaged_actions=config.val.with_damaged_actions,
        retain_every_n_frames_in_test=config.val.retain_every_n_frames_in_test,
        scale_bbox=config.val.scale_bbox,
        kind=config.kind,
        undistort_images=config.val.undistort_images,
        ignore_cameras=config.val.ignore_cameras,
        crop=config.val.crop,
        erase=config.val.erase,
        rank=rank,
        world_size=world_size,
        data_format=config.dataset.data_format,
        depth_format=aux_format,
        flow_format=flow_format,
        frame=args.frame
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.val.batch_size,
        shuffle=config.val.shuffle,
        # collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.val.randomize_n_views,
        #                                          min_n_views=config.val.min_n_views,
        #                                          max_n_views=config.val.max_n_views),
        num_workers=config.val.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, train_sampler, val_dataset.dist_size


def setup_dataloaders(config, is_train=True, distributed_train=False, rank = None, world_size=None):
    if config.dataset.kind == 'human36m':
        train_dataloader, val_dataloader, train_sampler, dist_size = setup_human36m_dataloaders(config, is_train, distributed_train, rank, world_size)
        _, whole_val_dataloader, _, _ = setup_human36m_dataloaders(config, is_train, distributed_train)
    else:
        raise NotImplementedError("Unknown dataset: {}".format(config.dataset.kind))
    
    return train_dataloader, val_dataloader, train_sampler, whole_val_dataloader, dist_size


def setup_experiment(config, model_name, is_train=True):
    prefix = "" if is_train else "eval_"

    if config.title:
        experiment_title = config.title + "_" + model_name
    else:
        experiment_title = model_name

    experiment_title = "ConPose"
    experiment_title = prefix + experiment_title

    experiment_name = '{}@{}'.format(experiment_title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    print("Experiment name: {}".format(experiment_name))

    experiment_dir = os.path.join(config.logdir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))

    # tensorboard
    writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

    # dump config to tensorboard
    writer.add_text(misc.config_to_str(config), "config", 0)

    return experiment_dir, writer

def one_epoch_full(model, criterion, loss_depth, optimizer, config, dataloader, device, epoch, n_iters_total=0, is_train=True, lr=None, mean_and_std=None, limb_length=None, caption='', master=False, experiment_dir=None, writer=None, whole_val_dataloader=None, dist_size=None, loss_depth_order=None):
    accum_iter = getattr(config.train, "accum_iter", 4)
    name = "train" if is_train else "val"
    model_type = config.model.name

    if is_train:
        epoch_loss_3d_train = 0
        N = 0

        if config.model.backbone.fix_weights:
            model_ref = model.module if hasattr(model, "module") else model
            model_ref.backbone.eval()
            model_ref.Lifting_net.train()
        # if config.model.backbone.fix_depth_weights:
        #     # model.module.depth_anything.eval()
        #     model.module.Lifting_net.train()
        else:
            model.train()
    else:
        model.eval()

    metric_dict = defaultdict(list)

    results = defaultdict(list)

    # used to turn on/off gradients
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    batch_idx = 0
    # count=0
    with grad_context():
        if config.model.name in DEPTH_FLOW_MODELS:
            aux_mode = "depth_flow"
        elif config.model.name in FLOW_MODELS:
            aux_mode = "flow"
        else:
            aux_mode = "depth"
        prefetcher = dataset_utils.data_prefetcher(
            dataloader,
            device,
            is_train,
            config.val.flip_test,
            aux_mode=aux_mode,
            flow_clip=getattr(config.dataset, "flow_clip", 20.0),
            flow_norm=getattr(config.dataset, "flow_norm", None),
        )

        # for iter_i, batch in iterator:
        batch = prefetcher.next()
        while batch is not None:
            # measure data loading time
            end = time.time()
            data_time = time.time() - end

            # images_batch, keypoints_3d_gt, keypoints_2d_batch_cpn, keypoints_2d_batch_cpn_crop, features_list_batch = batch
            if aux_mode == "depth_flow":
                images_batch, keypoints_3d_gt, keypoints_2d_batch_cpn, keypoints_2d_batch_cpn_crop, depth_images_batch, flow_images_batch = batch
            else:
                images_batch, keypoints_3d_gt, keypoints_2d_batch_cpn, keypoints_2d_batch_cpn_crop, aux_images_batch = batch
            
            if (not is_train) and config.val.flip_test:
                if aux_mode == "depth_flow":
                    keypoints_3d_pred, joint_depth, s = model(images_batch[:, 0], keypoints_2d_batch_cpn[:, 0], keypoints_2d_batch_cpn_crop[:, 0].clone(), depth_images_batch[:, 0], flow_images_batch[:, 0])
                    keypoints_3d_pred_flip, _, _ = model(images_batch[:, 1], keypoints_2d_batch_cpn[:, 1], keypoints_2d_batch_cpn_crop[:, 1].clone(), depth_images_batch[:, 1], flow_images_batch[:, 1])
                else:
                    keypoints_3d_pred, joint_depth, s = model(images_batch[:, 0], keypoints_2d_batch_cpn[:, 0], keypoints_2d_batch_cpn_crop[:, 0].clone(), aux_images_batch[:, 0])
                    keypoints_3d_pred_flip, _, _ = model(images_batch[:, 1], keypoints_2d_batch_cpn[:, 1], keypoints_2d_batch_cpn_crop[:, 1].clone(), aux_images_batch[:, 1])
                keypoints_3d_pred_flip[:, :, :, 0] *= -1
                keypoints_3d_pred_flip[:, :, joints_left + joints_right] = keypoints_3d_pred_flip[:, :, joints_right + joints_left]
                keypoints_3d_pred = torch.mean(torch.cat((keypoints_3d_pred, keypoints_3d_pred_flip), dim=1), dim=1,
                                                keepdim=True)
                del keypoints_3d_pred_flip


            else:    
                if aux_mode == "depth_flow":
                    keypoints_3d_pred, joint_depth, s = model(images_batch, keypoints_2d_batch_cpn, keypoints_2d_batch_cpn_crop, depth_images_batch, flow_images_batch)
                elif model_type in FLOW_UTILITY_MODELS and is_train:
                    keypoints_3d_pred, joint_depth, s = model(
                        images_batch,
                        keypoints_2d_batch_cpn,
                        keypoints_2d_batch_cpn_crop,
                        aux_images_batch,
                        keypoints_3d_gt=keypoints_3d_gt,
                    )
                else:
                    keypoints_3d_pred, joint_depth, s = model(images_batch, keypoints_2d_batch_cpn, keypoints_2d_batch_cpn_crop, aux_images_batch)

            n_joints = keypoints_3d_pred.shape[1]

            # calculate loss
            total_loss = 0.0
            loss = criterion(keypoints_3d_pred, keypoints_3d_gt)
            if model_type in ["DepthGuidedPosePoseAlign", "DepthGuidedPoseOARDG"]:
                loss_d = loss_depth(joint_depth, keypoints_3d_gt)
                total_loss += (loss + loss_d * getattr(config.loss, "lambda_layout", 0.01))
                metric_dict['layout_loss'].append(loss_d.item())
                for diag_name, diag_value in posealign_diagnostics(joint_depth, s, keypoints_3d_gt).items():
                    metric_dict[diag_name].append(diag_value)
            elif model_type in DLST_ORDER_MODELS:
                loss_d = loss_depth(joint_depth, keypoints_3d_gt)
                total_loss += (loss + loss_d * getattr(config.loss, "lambda_order", 0.001))
                metric_dict['order_loss'].append(loss_d.item())
                order_margin = getattr(config.loss, "order_margin", 0.05)
                for diag_name, diag_value in dlst_diagnostics(joint_depth, s, keypoints_3d_gt, order_margin).items():
                    metric_dict[diag_name].append(diag_value)
            elif model_type in ["DepthGuidedPoseUDEDLST", "DepthGuidedPoseDLSTUDE"]:
                # joint_depth = (coarse_depth, uncer) from UDE; s = (rel_depth, layer_assign) from DLST
                coarse_depth, uncer = joint_depth
                rel_depth, layer_assign = s
                loss_ude = loss_depth(coarse_depth, keypoints_3d_gt[..., -1:].squeeze(1), uncer)
                loss_order = loss_depth_order(rel_depth, keypoints_3d_gt)
                lambda_ude = getattr(config.loss, "lambda_ude", 0.00001)
                lambda_order = getattr(config.loss, "lambda_order", 0.001)
                total_loss += (loss + loss_ude * lambda_ude + loss_order * lambda_order)
                metric_dict['ude_loss'].append(loss_ude.item())
                metric_dict['order_loss'].append(loss_order.item())
                order_margin = getattr(config.loss, "order_margin", 0.05)
                for diag_name, diag_value in dlst_diagnostics(rel_depth, layer_assign, keypoints_3d_gt, order_margin).items():
                    metric_dict[diag_name].append(diag_value)
            elif model_type in FLOW_MODELS:
                total_loss += loss
                if model_type in FLOW_UTILITY_MODELS and isinstance(joint_depth, dict):
                    utility_loss = joint_depth.get("utility_loss")
                    if utility_loss is not None:
                        utility_weight = getattr(config.model.poseformer, "cfuafs_utility_weight", 0.05)
                        total_loss += utility_loss * utility_weight
                        metric_dict['cfuafs_utility_loss'].append(utility_loss.item())
                        metric_dict['cfuafs_utility_weight'].append(float(utility_weight))
                    for metric_name, metric_value in joint_depth.items():
                        if metric_name == "utility_loss":
                            continue
                        if isinstance(metric_value, (float, int)):
                            metric_dict[metric_name].append(float(metric_value))
            else:
                loss_d = loss_depth(joint_depth, keypoints_3d_gt[...,-1:].squeeze(1), s)
                total_loss += (loss + loss_d * 0.00001)
            metric_dict[config.loss.criterion].append(loss.item())

            metric_dict['total_loss'].append(total_loss.item())

            if is_train:
                epoch_loss_3d_train += keypoints_3d_gt.shape[0] * loss.item()
                N += keypoints_3d_gt.shape[0]

                if not torch.isnan(total_loss):
                    total_loss = total_loss  / accum_iter
                    total_loss.backward()
                    if ((batch_idx + 1) % accum_iter == 0):
                        optimizer.step()
                        optimizer.zero_grad()
                    batch_idx+=1
                    

                    if config.loss.grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.loss.grad_clip / config.train.Lifting_net_lr)

            # save answers for evalulation
            if not is_train:
                results['keypoints_gt'].append(keypoints_3d_gt.detach())    # (b, 17, 3)
                results['keypoints_3d'].append(keypoints_3d_pred.detach())    # (b, 17, 3)

            batch = prefetcher.next()

    if writer is not None and master and metric_dict:
        prefix = "train" if is_train else "val"
        step = epoch if epoch is not None else n_iters_total
        for metric_name, metric_values in metric_dict.items():
            if metric_values:
                writer.add_scalar("{}/{}".format(prefix, metric_name), float(np.mean(metric_values)), step)

    if master and metric_dict:
        diag_names = [
            "order_loss",
            "ude_loss",
            "dlst_rel_sign_acc",
            "dlst_assign_entropy",
            "dlst_layer_usage_min",
            "dlst_layer_usage_max",
            "cfuafs_utility_loss",
            "cfuafs_utility_top1",
            "cfuafs_weight_entropy",
        ]
        diag_parts = []
        for metric_name in diag_names:
            metric_values = metric_dict.get(metric_name)
            if metric_values:
                diag_parts.append("{} {:.6f}".format(metric_name, float(np.mean(metric_values))))
        if diag_parts:
            prefix = "train" if is_train else "val"
            diag_line = "[{} diagnostics] {}".format(prefix, " ".join(diag_parts))
            print(diag_line)
            if experiment_dir is not None:
                with open(os.path.join(experiment_dir, "out.txt"), "a") as log_file:
                    log_file.write(diag_line + "\n")

    if is_train:
        return epoch_loss_3d_train / N

    # calculate evaluation metrics
    if not is_train:
        if dist_size is not None:
            # term_list = ['keypoints_gt', 'keypoints_3d', 'proj_matricies_batch', 'indexes']
            term_list = ['keypoints_gt', 'keypoints_3d']
            for term in term_list:
                # results[term] = np.concatenate(results[term])
                results[term] = torch.cat(results[term])
                buffer = [torch.zeros(dist_size[-1], *results[term].shape[1:], device=device) for i in range(len(dist_size))]
                scatter_tensor = torch.zeros_like(buffer[0])
                scatter_tensor[:results[term].shape[0]] = results[term]
                # scatter_tensor[:results[term].shape[0]] = torch.tensor(results[term]).cuda()
                torch.distributed.all_gather(buffer, scatter_tensor)
                results[term] = torch.cat([tensor[:n] for tensor, n in zip(buffer, dist_size)], dim = 0)#.cpu().numpy()

    if master:
        if not is_train:
            if dist_size is None:
                results['keypoints_gt'] = torch.cat(results['keypoints_gt'])
                results['keypoints_3d'] = torch.cat(results['keypoints_3d'])
                print('evaluating....')
                result = dataloader.dataset.evaluate(results['keypoints_gt'], results['keypoints_3d'], None, config)
            else:
                result = whole_val_dataloader.dataset.evaluate(results['keypoints_gt'], results['keypoints_3d'], None, config)
            return result


def init_distributed(args):
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 1:
        return False

    torch.cuda.set_device(args.local_rank)

    assert os.environ["MASTER_PORT"], "set the MASTER_PORT variable or use pytorch launcher"
    assert os.environ["RANK"], "use pytorch launcher and explicityly state the rank of the process"

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    # torch.distributed.init_process_group(backend="gloo", init_method="env://")

    return True


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def main(args):
    is_distributed = init_distributed(args)

    master = True
    if is_distributed and os.environ["RANK"]:
        master = int(os.environ["RANK"]) == 0
        rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    else:
        rank = world_size = None

    if is_distributed:
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device("cuda:0")

    # config.train.n_iters_per_epoch = config.train.n_objects_per_epoch // config.train.batch_size   
    config.train.n_iters_per_epoch = None  

    # Backbone-specific configurations
    if args.backbone == 'hrnet_32':
        # Default setting
        config.model.poseformer.base_dim = 32

    elif args.backbone == 'hrnet_48':
        # Override the default setting
        config.model.backbone.checkpoint = 'data/pretrained/coco/pose_hrnet_w48_256x192.pth'
        config.model.backbone.STAGE2.NUM_CHANNELS = [48, 96]
        config.model.backbone.STAGE3.NUM_CHANNELS = [48, 96, 192]
        config.model.backbone.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
        config.model.poseformer.base_dim = 48                   

    if config.model.name == "DepthGuidedPoseRDGACMFR":
        model_cls = DepthGuidedPoseRDGACMFR
    elif config.model.name == "DepthGuidedPosePoseAlign":
        model_cls = DepthGuidedPosePoseAlign
    elif config.model.name == "DepthGuidedPoseOARDG":
        model_cls = DepthGuidedPoseOARDG
    elif config.model.name == "DepthGuidedPoseDLST":
        model_cls = DepthGuidedPoseDLST
    elif config.model.name == "DepthGuidedPoseDLSTCMFM":
        model_cls = DepthGuidedPoseDLSTCMFM
    elif config.model.name == "DepthFlowPoseDLST":
        model_cls = DepthFlowPoseDLST
    elif config.model.name == "DepthGuidedPoseUDEDLST":
        model_cls = DepthGuidedPoseUDEDLST
    elif config.model.name == "DepthGuidedPoseDLSTUDE":
        model_cls = DepthGuidedPoseDLSTUDE
    elif config.model.name == "RGBFlowPoseSingle":
        model_cls = RGBFlowPoseSingle
    elif config.model.name == "RGBFlowPoseCAPF":
        model_cls = RGBFlowPoseCAPF
    elif config.model.name == "RGBFlowPoseMFCE":
        model_cls = RGBFlowPoseMFCE
    elif config.model.name == "RGBFlowPoseMFCESeparate":
        model_cls = RGBFlowPoseMFCESeparate
    elif config.model.name == "RGBFlowPoseJGMS":
        model_cls = RGBFlowPoseJGMS
    elif config.model.name == "RGBFlowPoseCADS":
        model_cls = RGBFlowPoseCADS
    elif config.model.name == "RGBFlowPoseRRMS":
        model_cls = RGBFlowPoseRRMS
    elif config.model.name == "RGBFlowPoseMRRDCE":
        model_cls = RGBFlowPoseMRRDCE
    elif config.model.name == "RGBFlowPoseMFAS":
        model_cls = RGBFlowPoseMFAS
    elif config.model.name == "RGBFlowPoseAFFS":
        model_cls = RGBFlowPoseAFFS
    elif config.model.name == "RGBFlowPoseMES":
        model_cls = RGBFlowPoseMES
    elif config.model.name == "RGBFlowPoseAOFS":
        model_cls = RGBFlowPoseAOFS
    elif config.model.name == "RGBFlowPoseLMRS":
        model_cls = RGBFlowPoseLMRS
    elif config.model.name == "RGBFlowPoseCMFF":
        model_cls = RGBFlowPoseCMFF
    elif config.model.name == "RGBFlowPoseMCES":
        model_cls = RGBFlowPoseMCES
    elif config.model.name == "RGBFlowPoseCFUAFS":
        model_cls = RGBFlowPoseCFUAFS
    elif config.model.name == "RGBFlowPoseUAFS":
        model_cls = RGBFlowPoseUAFS
    else:
        model_cls = DepthGuidedPose
    model = model_cls(config, device)

    # experiment
    experiment_dir, writer = None, None
    if master:
        experiment_dir, writer = setup_experiment(config, type(model).__name__, is_train=not args.eval)
        shutil.copy('mvn/models/DGPose.py', experiment_dir)
        shutil.copy('mvn/models/DGLifting.py', experiment_dir)
        if config.model.name == "DepthGuidedPoseRDGACMFR":
            shutil.copy('mvn/models/DGPose_rdga_cmfr.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rdga_cmfr.py', experiment_dir)
        elif config.model.name == "DepthGuidedPosePoseAlign":
            shutil.copy('mvn/models/DGPose_posealign.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_posealign.py', experiment_dir)
        elif config.model.name == "DepthGuidedPoseOARDG":
            shutil.copy('mvn/models/DGPose_oardg.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_oardg.py', experiment_dir)
        elif config.model.name == "DepthGuidedPoseDLST":
            shutil.copy('mvn/models/DGPose_dlst.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_dlst.py', experiment_dir)
        elif config.model.name == "DepthGuidedPoseDLSTCMFM":
            shutil.copy('mvn/models/DGPose_dlst_cmfm.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_dlst_cmfm.py', experiment_dir)
        elif config.model.name == "DepthFlowPoseDLST":
            shutil.copy('mvn/models/DGPose_depthflow_dlst.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_depthflow_dlst.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_dlst.py', experiment_dir)
        elif config.model.name == "DepthGuidedPoseUDEDLST":
            shutil.copy('mvn/models/DGPose_ude_dlst.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_ude_dlst.py', experiment_dir)
        elif config.model.name == "DepthGuidedPoseDLSTUDE":
            shutil.copy('mvn/models/DGPose_dlst_ude.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_dlst_ude.py', experiment_dir)
        elif config.model.name == "RGBFlowPoseSingle":
            shutil.copy('mvn/models/DGPose_rgbflow_single.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_single.py', experiment_dir)
        elif config.model.name == "RGBFlowPoseCAPF":
            shutil.copy('mvn/models/DGPose_rgbflow_capf.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_capf.py', experiment_dir)
        elif config.model.name == "RGBFlowPoseMFCE":
            shutil.copy('mvn/models/DGPose_rgbflow_mfce.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_mfce.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_capf.py', experiment_dir)
        elif config.model.name == "RGBFlowPoseMFCESeparate":
            shutil.copy('mvn/models/DGPose_rgbflow_mfce_separate.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_mfce_separate.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_capf.py', experiment_dir)
        elif config.model.name == "RGBFlowPoseJGMS":
            shutil.copy('mvn/models/DGPose_rgbflow_jgms.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_jgms.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_capf.py', experiment_dir)
        elif config.model.name == "RGBFlowPoseCADS":
            shutil.copy('mvn/models/DGPose_rgbflow_cads.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_cads.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_capf.py', experiment_dir)
        elif config.model.name == "RGBFlowPoseRRMS":
            shutil.copy('mvn/models/DGPose_rgbflow_rrms.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_rrms.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_capf.py', experiment_dir)
        elif config.model.name == "RGBFlowPoseMRRDCE":
            shutil.copy('mvn/models/DGPose_rgbflow_mrrdce.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_mrrdce.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_capf.py', experiment_dir)
        elif config.model.name == "RGBFlowPoseMFAS":
            shutil.copy('mvn/models/DGPose_rgbflow_mfas.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_mfas.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_capf.py', experiment_dir)
        elif config.model.name == "RGBFlowPoseAFFS":
            shutil.copy('mvn/models/DGPose_rgbflow_affs.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_affs.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_capf.py', experiment_dir)
        elif config.model.name == "RGBFlowPoseMES":
            shutil.copy('mvn/models/DGPose_rgbflow_mes.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_mes.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_capf.py', experiment_dir)
        elif config.model.name == "RGBFlowPoseAOFS":
            shutil.copy('mvn/models/DGPose_rgbflow_aofs.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_aofs.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_capf.py', experiment_dir)
        elif config.model.name == "RGBFlowPoseLMRS":
            shutil.copy('mvn/models/DGPose_rgbflow_lmrs.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_lmrs.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_capf.py', experiment_dir)
        elif config.model.name == "RGBFlowPoseCMFF":
            shutil.copy('mvn/models/DGPose_rgbflow_cmff.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_cmff.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_capf.py', experiment_dir)
        elif config.model.name == "RGBFlowPoseMCES":
            shutil.copy('mvn/models/DGPose_rgbflow_mces.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_mces.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_capf.py', experiment_dir)
        elif config.model.name == "RGBFlowPoseCFUAFS":
            shutil.copy('mvn/models/DGPose_rgbflow_cfuafs.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_cfuafs.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_capf.py', experiment_dir)
        elif config.model.name == "RGBFlowPoseUAFS":
            shutil.copy('mvn/models/DGPose_rgbflow_uafs.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_uafs.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_cfuafs.py', experiment_dir)
            shutil.copy('mvn/models/DGLifting_rgbflow_capf.py', experiment_dir)
        shutil.copy('train_bnn.py', experiment_dir)
        # shutil.copy('mvn/models/pose_cformer.py', experiment_dir)

    print("args: {}".format(args))
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))

    if config.model.backbone.init_weights:
        # Load HRNet
        if args.backbone in ['hrnet_32', 'hrnet_48']:
            ret = model.backbone.load_state_dict(
                torch.load(config.model.backbone.checkpoint, map_location='cpu'),
                strict=False
            )
        print(ret)
        print("Loading backbone from {}".format(config.model.backbone.checkpoint))

    if args.init_checkpoint_path:
        raw = torch.load(args.init_checkpoint_path, map_location='cpu')
        checkpoint = raw['model'] if 'model' in raw else raw
        checkpoint = {key.replace("module.", ""): value for key, value in checkpoint.items()}
        model_state = model.state_dict()
        init_state = {}
        skipped = []
        for key, value in checkpoint.items():
            if key in model_state and tuple(value.shape) == tuple(model_state[key].shape):
                init_state[key] = value
            else:
                skipped.append(key)
        model_state.update(init_state)
        model.load_state_dict(model_state, strict=True)
        print(
            "Initialized {} tensors from {} (skipped {}).".format(
                len(init_state), args.init_checkpoint_path, len(skipped)
            )
        )

    if args.freeze_except:
        keep_keywords = [item.strip() for item in args.freeze_except.split(',') if item.strip()]
        for name, parameter in model.named_parameters():
            parameter.requires_grad = any(keyword in name for keyword in keep_keywords)
        trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        print("Freeze except {} -> trainable params: {}".format(keep_keywords, trainable_params))

    if args.eval:
        ckpt_path = args.checkpoint_path if args.checkpoint_path else "./checkpoints/best_epoch.bin"
        raw = torch.load(ckpt_path, map_location='cpu')
        checkpoint = raw['model'] if 'model' in raw else raw
        for key in list(checkpoint.keys()):
            new_key = key.replace("module.", "")
            checkpoint[new_key] = checkpoint.pop(key)

        ret = model.load_state_dict(checkpoint, strict=False)
        print(ret)
        print("Loading checkpoint from {}".format(ckpt_path))

    # sync bn in multi-gpus
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)

    # criterion
    criterion_class = {
        "MPJPE": MPJPE,
        "MSE": KeypointsMSELoss,
        "MSESmooth": KeypointsMSESmoothLoss,
        "MAE": KeypointsMAELoss
    }[config.loss.criterion]
    loss_depth_order = None
    if config.model.name in DLST_ORDER_MODELS:
        loss_depth = DepthOrderingLoss(
            margin=getattr(config.loss, "order_margin", 0.05),
            temperature=getattr(config.loss, "order_temperature", 1.0),
            logit_scale=getattr(config.loss, "order_logit_scale", 4.0),
            auto_unit=getattr(config.loss, "order_auto_unit", True),
        )
    elif config.model.name in ["DepthGuidedPosePoseAlign", "DepthGuidedPoseOARDG"]:
        loss_depth = DepthLayoutLoss()
    elif config.model.name in ["DepthGuidedPoseUDEDLST", "DepthGuidedPoseDLSTUDE"]:
        # baseline UDE supervision (BNN) + DLST ordering supervision
        loss_depth = BNNLoss()
        loss_depth_order = DepthOrderingLoss(
            margin=getattr(config.loss, "order_margin", 0.05),
            temperature=getattr(config.loss, "order_temperature", 1.0),
            logit_scale=getattr(config.loss, "order_logit_scale", 4.0),
            auto_unit=getattr(config.loss, "order_auto_unit", True),
        )
    elif config.model.name in FLOW_MODELS:
        loss_depth = None
    else:
        loss_depth = BNNLoss()

    if config.loss.criterion == "MSESmooth":
        criterion = criterion_class(config.loss.mse_smooth_threshold).to(device)
    else:
        criterion = criterion_class().to(device)

    # optimizer
    opt_dict = None
    lr_schd_dict = None
    lr_dict = None
    lr = config.train.Lifting_net_lr
    lr_decay = config.train.Lifting_net_lr_decay
    if not args.eval:
        opt_dict = {}
        lr_schd_dict = {}
        lr_dict = {}
        
        param_dicts = [
        # {
        # 	"params":
        # 		[p for n, p in model.backbone.named_parameters() if p.requires_grad],
        # 	"lr": config.train.backbone_lr * 0.1,
        # },
        {
            "params":
                [p for n, p in model.Lifting_net.named_parameters() if p.requires_grad],
            "lr": config.train.Lifting_net_lr,
        },
        ]
        optimizer = optim.AdamW(param_dicts, weight_decay=0.1)

        
    # datasets
    if master:
        print("Loading data...")
    train_dataloader, val_dataloader, train_sampler, whole_val_dataloader, dist_size = setup_dataloaders(config, distributed_train=is_distributed, rank=rank, world_size=world_size)

    log_file_path = os.path.join(experiment_dir, "out.txt") if master else None
    if master:
        model_params = 0
        for parameter in model.Lifting_net.parameters():
            model_params += parameter.numel()
        print("Trainable parameter count: " + str(model_params))
        with open(log_file_path, 'a') as log_file:
            log_file.write("Trainable parameter count: [%d]\n" %  model_params)

    # multi-gpu
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank) # , find_unused_parameters=True

    if not args.eval:
        # train loop
        losses_3d_train = []
        min_loss = 100000
        n_iters_total_train, n_iters_total_val = 0, 0
        
        for epoch in range(config.train.n_epochs):
            errors_p1 = []
            errors_p2 = []
            start_time = time.time()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            epoch_loss_3d_train = one_epoch_full(model, criterion, loss_depth, optimizer, config, train_dataloader, device, epoch, n_iters_total=n_iters_total_train, is_train=True, lr=lr_dict, master=master, experiment_dir=experiment_dir, writer=writer, loss_depth_order=loss_depth_order)
            result = one_epoch_full(model, criterion, loss_depth, optimizer, config, val_dataloader, device, epoch, n_iters_total=n_iters_total_val, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer, whole_val_dataloader=whole_val_dataloader, dist_size=dist_size, loss_depth_order=loss_depth_order)

            losses_3d_train.append(epoch_loss_3d_train)

            if master:
                for k in result.keys():
                    errors_p1.append(result[k]['MPJPE'] * 1000)
                    errors_p2.append(result[k]['P_MPJPE'] * 1000)

                error_p1 = round(np.mean(errors_p1), 1)
                error_p2 = round(np.mean(errors_p2), 1)

                print('[%d] time %.2f lr %f 3d_train %f 3d_test_p1 %f 3d_test_p2 %f' % (
                    epoch + 1,
                    (time.time()-start_time) / 60.,
                    lr,
                    losses_3d_train[-1] * 1000,
                    error_p1,
                    error_p2))
                with open(log_file_path, 'a') as log_file:
                    log_file.write('[%d] time %.2f lr %f 3d_train %f 3d_test_p1 %f 3d_test_p2 %f\n' % (
                        epoch + 1,
                        (time.time() - start_time) / 60.,
                        lr,
                        losses_3d_train[-1] * 1000,
                        error_p1,
                        error_p2
                    ))

                if error_p1 < min_loss:
                    min_loss = error_p1
                    print("save best checkpoint")
                    with open(log_file_path, 'a') as log_file:
                        log_file.write("save best checkpoint \n")
                    torch.save({
                        'epoch': epoch + 1,
                        'lr': lr,
                        # 'random_state': train_generator.random_state(),
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict(),
                    }, os.path.join(experiment_dir, "checkpoints/best_epoch.bin"))

            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

    else:
        errors_p1 = []
        errors_p2 = []
        errors_vel = []
        dataloader = train_dataloader if args.eval_dataset == 'train' else val_dataloader
        result = one_epoch_full(model, criterion, loss_depth, None, config, val_dataloader, device, None, n_iters_total=0, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer, whole_val_dataloader=whole_val_dataloader, dist_size=dist_size, loss_depth_order=loss_depth_order)

        if master:
            for k in result.keys():
                print(k, "p1:", result[k]['MPJPE'] * 1000, "p2:", result[k]['P_MPJPE'] * 1000, "e_vel:", result[k]['MPJVE'] * 1000)
                errors_p1.append(result[k]['MPJPE'] * 1000)
                errors_p2.append(result[k]['P_MPJPE'] * 1000)
                # errors_p3.append(result[k]['N_MPJPE'] * 1000)
                errors_vel.append(result[k]['MPJVE'] * 1000)

            error_p1 = round(np.mean(errors_p1), 1)
            error_p2 = round(np.mean(errors_p2), 1)
            error_vel = round(np.mean(errors_vel), 2)
            print("avg", "p1:", error_p1, "p2:", error_p2, "MPJVE:", error_vel)

            print("Done.")

if __name__ == '__main__':
    args = parse_args()
    main(args)
