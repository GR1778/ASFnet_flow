import numpy as np
import unittest

from tools.diagnose_flow_mfce_sampling import (
    bilinear_sample_flow,
    dce_initial_offsets_px,
    diagnose_for_offsets,
)


class FlowMFCESamplingDiagnosisTest(unittest.TestCase):
    def test_dce_offsets_match_pixel_scale_and_grow_with_samples(self):
        offsets_4 = dce_initial_offsets_px(num_heads=4, num_samples=4, width=192, height=256)
        offsets_5 = dce_initial_offsets_px(num_heads=4, num_samples=5, width=192, height=256)

        self.assertEqual(offsets_4.shape, (4, 4, 2))
        self.assertEqual(offsets_5.shape, (4, 5, 2))
        self.assertGreater(np.max(np.abs(offsets_5[..., 0])), np.max(np.abs(offsets_4[..., 0])))
        self.assertGreater(np.max(np.abs(offsets_5[..., 1])), np.max(np.abs(offsets_4[..., 1])))

        # The fifth sample in ASFNet/MFCE reaches roughly 4.77px horizontally
        # and 6.37px vertically at the 192x256 crop resolution.
        self.assertTrue(np.isclose(np.max(np.abs(offsets_5[..., 0])), 4.771, atol=0.02))
        self.assertTrue(np.isclose(np.max(np.abs(offsets_5[..., 1])), 6.369, atol=0.02))

    def test_bilinear_sampler_uses_border_values_at_edges(self):
        flow = np.zeros((4, 4, 2), dtype=np.float32)
        flow[3, 3] = [2.0, -1.0]

        sampled = bilinear_sample_flow(flow, np.array([[3.0, 3.0]], dtype=np.float32))
        self.assertTrue(np.allclose(sampled[0], [2.0, -1.0]))

    def test_motion_boundary_can_make_multi_sampling_worse_than_center(self):
        flow = np.zeros((64, 64, 2), dtype=np.float32)
        flow[:, :32, 0] = 1.0
        flow[:, 32:, 0] = -1.0

        cur_cpn = np.array([[31.2, 32.0]], dtype=np.float32)
        gt_target = np.array([[1.0, 0.0]], dtype=np.float32)
        stats = np.zeros((1, 4), dtype=np.float32)
        offsets = dce_initial_offsets_px(num_heads=4, num_samples=5, width=64, height=64)

        row = diagnose_for_offsets(
            flow=flow,
            cur_cpn=cur_cpn,
            gt_target=gt_target,
            stats=stats,
            offsets_px=offsets,
            meta={"subject": 1, "action": 2, "subaction": 1, "camera_id": 0, "frame_id": 100},
            margin=0.05,
        )[0]

        self.assertLess(row["center_error"], 0.5)
        self.assertGreater(row["multi_mean_error"], row["center_error"])
        self.assertGreater(row["frac_flow_delta_gt_1px"], 0.0)


if __name__ == "__main__":
    unittest.main()
