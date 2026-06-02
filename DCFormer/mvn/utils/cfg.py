import yaml
from easydict import EasyDict as edict
import os

config = edict()

config.title = "human36m_single"
config.kind = "human36m"
config.azureroot = ""
config.logdir = "logs"
config.batch_output = False
config.vis_freq = 1000
config.vis_n_elements = 10
config.id = 600
config.frame = 1

# model definition
config.model = edict()
config.model.name = "DepthGuidedPose"
config.model.image_shape = [192, 256]
config.model.heatmap_shape = [96, 96]
config.model.heatmap_softmax = True
config.model.heatmap_multiplier = 100.0
config.model.init_weights = True
config.model.checkpoint = None

config.model.backbone = edict()
config.model.backbone.type = 'hrnet_32'
config.model.backbone.num_final_layer_channel = 17
config.model.backbone.num_joints = 17
config.model.backbone.num_layers = 152
config.model.backbone.init_weights = True
config.model.backbone.fix_weights = False
config.model.backbone.checkpoint = "data/pretrained/human36m/pose_hrnet_w32_256x192.pth"
# config.model.backbone.depth_checkpoint = 'depth_anything/checkpoint/depth_anything_vitl14.pth'
# config.model.backbone.fix_depth_weights = True
# config.model.backbone.init_depth_weights = True

# pose_hrnet related params
# config.model.backbone = edict()
config.model.backbone.NUM_JOINTS = 17
config.model.backbone.PRETRAINED_LAYERS = ['*']
config.model.backbone.STEM_INPLANES = 64
config.model.backbone.FINAL_CONV_KERNEL = 1

config.model.backbone.STAGE2 = edict()
config.model.backbone.STAGE2.NUM_MODULES = 1
config.model.backbone.STAGE2.NUM_BRANCHES = 2
config.model.backbone.STAGE2.NUM_BLOCKS = [4, 4]
config.model.backbone.STAGE2.NUM_CHANNELS = [32, 64]
# config.model.backbone.STAGE2.NUM_CHANNELS = [48, 96]
config.model.backbone.STAGE2.BLOCK = 'BASIC'
config.model.backbone.STAGE2.FUSE_METHOD = 'SUM'

config.model.backbone.STAGE3 = edict()
# config.model.backbone.STAGE3.NUM_MODULES = 1
config.model.backbone.STAGE3.NUM_MODULES = 4
config.model.backbone.STAGE3.NUM_BRANCHES = 3
config.model.backbone.STAGE3.NUM_BLOCKS = [4, 4, 4]
config.model.backbone.STAGE3.NUM_CHANNELS = [32, 64, 128]
# config.model.backbone.STAGE3.NUM_CHANNELS = [48, 96, 192]
config.model.backbone.STAGE3.BLOCK = 'BASIC'
config.model.backbone.STAGE3.FUSE_METHOD = 'SUM'

config.model.backbone.STAGE4 = edict()
# config.model.backbone.STAGE4.NUM_MODULES = 1
config.model.backbone.STAGE4.NUM_MODULES = 3
config.model.backbone.STAGE4.NUM_BRANCHES = 4
config.model.backbone.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
config.model.backbone.STAGE4.NUM_CHANNELS = [32, 64, 128, 256]
# config.model.backbone.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
config.model.backbone.STAGE4.BLOCK = 'BASIC'
config.model.backbone.STAGE4.FUSE_METHOD = 'SUM'

# pose_resnet related params
config.model.backbone.NUM_LAYERS = 50
config.model.backbone.DECONV_WITH_BIAS = False
config.model.backbone.NUM_DECONV_LAYERS = 3
config.model.backbone.NUM_DECONV_FILTERS = [256, 256, 256]
config.model.backbone.NUM_DECONV_KERNELS = [4, 4, 4]
config.model.backbone.FINAL_CONV_KERNEL = 1
config.model.backbone.PRETRAINED_LAYERS = ['*']

config.model.poseformer = edict()
config.model.poseformer.embed_dim_ratio = 128
config.model.poseformer.base_dim = 32
config.model.poseformer.depth = 4
config.model.poseformer.flow_num_heads = 4
config.model.poseformer.flow_num_samples = 5
config.model.poseformer.flow_local_radius_px = 12.0
config.model.poseformer.flow_residual_radius_px = 4.0
config.model.poseformer.flow_gate_init = -1.5
config.model.poseformer.flow_encoder_layers = 1
config.model.poseformer.mces_num_heads = 4
config.model.poseformer.mces_num_samples = 5
config.model.poseformer.mces_offset_scale = 0.125
config.model.poseformer.mces_consistency_init = 1.0
config.model.poseformer.cfuafs_num_samples = 5
config.model.poseformer.cfuafs_radius_px = 8.0
config.model.poseformer.cfuafs_center_bias = 4.0
config.model.poseformer.cfuafs_utility_tau = 0.01
config.model.poseformer.cfuafs_utility_weight = 0.05
config.model.poseformer.cfuafs_enable_utility = True
config.model.poseformer.aofs_kernel_size = 5
config.model.poseformer.aofs_radius_px = 8.0
config.model.poseformer.aofs_num_heads = 4
config.model.poseformer.lmrs_radii = [2, 4, 6, 8, 12, 16]
config.model.poseformer.lmrs_num_directions = 16
config.model.poseformer.lmrs_sigma_flow = 0.5
config.model.poseformer.lmrs_sigma_pos = 8.0
config.model.poseformer.lmrs_tau = 0.5
config.model.poseformer.lmrs_topk = 0
config.model.poseformer.dlst_num_depth_layers = 4
config.model.poseformer.dlst_num_heads = 8
config.model.poseformer.dlst_depth = 1
config.model.poseformer.dlst_mlp_ratio = 2.0
config.model.poseformer.dlst_drop_path = 0.0
config.model.poseformer.dlst_assignment_temperature = 1.0
config.model.poseformer.dlst_omega_temperature = 1.0
config.model.poseformer.dlst_depth_gate_init = 0.1
config.model.poseformer.use_cmfm = True
config.model.poseformer.cmff_depth = 2
config.model.poseformer.cmff_heads = 8
config.model.poseformer.cmfm_backend = "vmamba"
config.model.poseformer.cmfm_forward_type = "v05_noz"
config.model.poseformer.cmfm_initialize = "v0"
config.model.poseformer.cmfm_d_state = 1
config.model.poseformer.cmfm_d_conv = 3
config.model.poseformer.cmfm_expand = 2
config.model.poseformer.cmfm_conv_bias = False
config.model.poseformer.cmfm_eca_kernel = 3
config.model.poseformer.cmfm_init_scale = 0.1
config.model.poseformer.cmfm_drop_path = 0.1
config.model.poseformer.cmfm_keep_depth_token = False
config.model.poseformer.cmfm_residual_scale = 1.0
config.model.poseformer.pcfe_use_ssm = False
config.model.poseformer.pcfe_backend = "mamba"
config.model.poseformer.pcfe_d_state = 16
config.model.poseformer.pcfe_d_conv = 4
config.model.poseformer.pcfe_expand = 2
config.model.poseformer.ldsr = edict()
config.model.poseformer.ldsr.num_slots = 6
config.model.poseformer.ldsr.num_heads = 4
config.model.poseformer.ldsr.slot_iters = 2

config.model.poseformer.posealign = edict()
config.model.poseformer.posealign.num_anchors = 7
config.model.poseformer.posealign.num_heads = 4
config.model.poseformer.posealign.relation_temperature = 1.5
config.model.poseformer.posealign.near_threshold = 0.25
config.model.poseformer.posealign.depth_gate_temperature = 4.0

# RDGA-CMFR depth branch ablations.
# RDGA refines AMS depth tokens with relative-depth geometry attention.
# CMFR applies lightweight reliability-aware pose/depth rectification before fusion.
config.model.poseformer.rdga_cmfr = edict()
config.model.poseformer.rdga_cmfr.use_rdga = True
config.model.poseformer.rdga_cmfr.use_cmfr = True
config.model.poseformer.rdga_cmfr.rdga_layers = 2
config.model.poseformer.rdga_cmfr.rdga_heads = 4
config.model.poseformer.rdga_cmfr.rdga_mlp_ratio = 2.0
config.model.poseformer.rdga_cmfr.cmfr_reduction = 1
config.model.poseformer.rdga_cmfr.cmfr_lambda_c = 0.5
config.model.poseformer.rdga_cmfr.cmfr_lambda_t = 0.5
config.model.poseformer.rdga_cmfr.cmfr_zero_init = True

# DOGA: Depth Ordering Graph Attention (replaces UDE).
# Switches control sub-component ablations; keep all True for the full DOGA.
# `num_blocks` / `mlp_ratio` control OBCA stack capacity. `use_geom_prior`
# injects 2D position + H36M skeleton adjacency into the PwOP. `use_aux_abs`
# adds a coarse absolute-depth regression head as safety-net supervision;
# `lambda_abs` weights its smooth-L1 term.
config.model.doga = edict()
config.model.doga.enabled = True
config.model.doga.use_sinkhorn = True
config.model.doga.use_obca = True
config.model.doga.use_rape = True
config.model.doga.sinkhorn_iter = 20
config.model.doga.num_blocks = 4
config.model.doga.mlp_ratio = 5.0
config.model.doga.use_geom_prior = True
config.model.doga.use_aux_abs = True
config.model.doga.lambda_abs = 0.1

# loss related params
config.loss = edict()
config.loss.criterion = "MAE"
config.loss.mse_smooth_threshold = 0
config.loss.grad_clip = 0
config.loss.scale_keypoints_3d = 0.1
config.loss.use_volumetric_ce_loss = True
config.loss.volumetric_ce_loss_weight = 0.01
config.loss.use_global_attention_loss = True
config.loss.global_attention_loss_weight = 1000000
config.loss.lambda_layout = 0.01
config.loss.lambda_ude = 0.00001
config.loss.lambda_order = 0.001
config.loss.order_margin = 0.05
config.loss.order_temperature = 1.0
config.loss.order_logit_scale = 4.0
config.loss.order_auto_unit = True

# dataset related params
config.dataset = edict()
config.dataset.kind = "human36m"
config.dataset.data_format = ''
config.dataset.transfer_cmu_to_human36m = False
# config.dataset.root = "data/human36m/processed/"
config.dataset.root = "../H36M-Toolbox/images_crop/"
config.dataset.extra_root = "data/human36m/extra"
config.dataset.train_labels_path = "data/human36m/extra/human36m-multiview-labels-GTbboxes.npy"
config.dataset.val_labels_path = "data/human36m/extra/human36m-multiview-labels-GTbboxes.npy"
config.dataset.depth_image_path = "../H36M-Toolbox/depth_images_RGB/"
config.dataset.depth_format = "image"
config.dataset.flow_image_path = "../H36M-Toolbox/flow_images_float/"
config.dataset.flow_format = "flow_npy"
config.dataset.flow_clip = 20.0
config.dataset.flow_norm = None
config.dataset.train_dataset = "multiview_human36m"
config.dataset.val_dataset = "human36m"

# train related params
config.train = edict()
config.train.n_objects_per_epoch = 15000
config.train.n_epochs = 9999
config.train.n_iters_per_epoch = 5000
config.train.batch_size = 3
config.train.accum_iter = 1
config.train.optimizer = 'Adam'
config.train.backbone_lr = 0.0001
config.train.backbone_lr_step = [1000]
config.train.backbone_lr_factor = 0.1
config.train.Lifting_net_lr = 0.001
config.train.Lifting_net_lr_decay = 0.99
config.train.with_damaged_actions = True
config.train.undistort_images = True
config.train.scale_bbox = 1.0
config.train.ignore_cameras = []
config.train.crop = True
config.train.erase = False
config.train.shuffle = True
config.train.randomize_n_views = True
config.train.min_n_views = 1
config.train.max_n_views = 1
config.train.num_workers = 8
config.train.limb_length_path = "data/human36m/extra/mean_and_std_limb_length.h5"
config.train.pred_results_path = "data/pretrained/human36m/human36m_alg_10-04-2019/checkpoints/0060/results/train.pkl"

# val related params
config.val = edict()
config.val.flip_test = True
config.val.batch_size = 6
config.val.with_damaged_actions = True
config.val.undistort_images = True
config.val.scale_bbox = 1.0
config.val.ignore_cameras = []
config.val.crop = True
config.val.erase = False
config.val.shuffle = False
config.val.randomize_n_views = True
config.val.min_n_views = 1
config.val.max_n_views = 1
config.val.num_workers = 10
config.val.retain_every_n_frames_in_test = 1
config.val.limb_length_path = "data/human36m/extra/mean_and_std_limb_length.h5"
config.val.pred_results_path = "data/pretrained/human36m/human36m_alg_10-04-2019/checkpoints/0060/results/val.pkl"

# def update_dict(v, cfg):
#     for kk, vv in v.items():
#         if kk in cfg:
#             if isinstance(vv, dict):
#                 print("vv",vv)
#                 print("cfg[kk]",cfg[kk])
#                 update_dict(vv, cfg[kk])
#             else:
#                 cfg[kk] = vv
#         else:
#             raise ValueError("{} not exist in cfg.py".format(kk))
def update_dict(v, cfg):
    for kk, vv in v.items():
        if kk in cfg and isinstance(cfg[kk], dict):
            if isinstance(vv, dict):
                update_dict(vv, cfg[kk])
            else:
                cfg[kk] = vv
        elif kk in cfg:
            cfg[kk] = vv
        else:
            raise ValueError("{} not exist in cfg.py".format(kk))


def update_config(path):
    exp_config = None
    with open(path) as fin:
        exp_config = edict(yaml.safe_load(fin))
        update_dict(exp_config, config)


def handle_azureroot(config_dict, azureroot):
    for key in config_dict.keys():
        if isinstance(config_dict[key], str):
            if config_dict[key].startswith('data/'):
                config_dict[key] = os.path.join(azureroot, config_dict[key])
        elif isinstance(config_dict[key], dict):
            handle_azureroot(config_dict[key], azureroot)


def update_dir(azureroot, logdir):
    config.azureroot = azureroot
    config.logdir = os.path.join(config.azureroot, logdir)
    if config.model.checkpoint != None and not config.model.checkpoint.startswith('data/'):
        config.model.checkpoint = os.path.join(config.azureroot, config.model.checkpoint)
    handle_azureroot(config, config.azureroot)   

   
