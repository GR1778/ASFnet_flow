# ASFnet Pose Alignment Variant

Official ASFnet-style launch:

```powershell
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2345 train.py --config <PATH-TO-CONFIG> --logdir ./logs
```

Our pose-alignment config:

```powershell
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2345 train.py --config experiments/human36m/human36m_single_posealign.yaml --logdir ./logs
```

Eval:

```powershell
python -m torch.distributed.launch --nproc_per_node=1 --master_port=2345 train.py --config experiments/human36m/human36m_single_posealign.yaml --logdir ./logs --eval
```
