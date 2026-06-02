from mvn.models.DGLifting_rgbflow_cfuafs import RGBFlowCFUAFSLifting


class RGBFlowUAFSLifting(RGBFlowCFUAFSLifting):
    """
    Flow-only bounded adaptive sampling for from-scratch training.

    This uses the same bounded candidate sampler as CFUAFS but disables the
    counterfactual/P1-utility auxiliary path. The RGB DCE branch and downstream
    fusion remain unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfuafs_enable_utility = False

    def forward(self, keypoints_2d, ref, flow_images, features_list_hr, keypoints_3d_gt=None):
        keypoints_3d, _ = super().forward(
            keypoints_2d,
            ref,
            flow_images,
            features_list_hr,
            keypoints_3d_gt=None,
        )
        return keypoints_3d, None
