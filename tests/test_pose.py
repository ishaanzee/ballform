import numpy as np
import pytest
from ultralytics.data.augment import LetterBox

from app.pose import EXPORT_SHAPES, CoreMLPose, decode, letterbox, letterbox_shape


@pytest.mark.parametrize(("height", "width", "size"), [(1080, 1920, 1280), (1080, 1152, 960), (720, 1280, 1280),
                                                       (720, 768, 960), (1080, 1440, 1280)])
def test_letterbox_matches_ultralytics_rect_prediction(height, width, size):
    image = np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)
    expected = LetterBox((size, size), auto=True, stride=32)(image=image)
    assert letterbox_shape(height, width, size) == expected.shape[:2]
    padded, _, _ = letterbox(image, expected.shape[:2])
    assert np.array_equal(padded, expected)


def test_16_by_9_frames_and_side_crops_use_the_exported_shapes():
    for width, height in ((1920, 1080), (1280, 720), (3840, 2160)):
        crop_width = round(width * .6)
        assert letterbox_shape(height, width, 1280) == EXPORT_SHAPES[1280]
        assert letterbox_shape(height, crop_width, 960) == EXPORT_SHAPES[960]
    assert letterbox_shape(1080, 1440, 1280) != EXPORT_SHAPES[1280]


def test_decode_suppresses_overlaps_and_maps_back_to_source_pixels():
    # Two overlapping people and one low-confidence anchor in a 20 px-padded input.
    raw = np.zeros((1, 56, 3), dtype=np.float32)
    raw[0, :5, 0] = (100, 120, 40, 80, .9)
    raw[0, :5, 1] = (102, 121, 40, 80, .8)
    raw[0, :5, 2] = (300, 300, 40, 80, .2)
    raw[0, 5::3, 0] = 110
    raw[0, 6::3, 0] = 140
    raw[0, 7::3, 0] = .7
    boxes, scores, keypoints = decode(raw, gain=.5, pad=(0, 20), height=1000, width=1000)
    assert scores.tolist() == pytest.approx([.9])
    assert boxes[0].tolist() == pytest.approx([160, 120, 240, 280])
    assert keypoints.shape == (1, 17, 3)
    assert keypoints[0, 0].tolist() == pytest.approx([220, 240, .7])


def test_coreml_pose_uses_torch_for_shapes_it_was_not_exported_for():
    class Fallback:
        calls = []

        def infer(self, image, size):
            self.calls.append((image.shape, size))
            return None

    pose = CoreMLPose.__new__(CoreMLPose)
    pose.fallback, pose.fallback_calls = Fallback(), 0
    pose.sessions = {1280: (None, "images", EXPORT_SHAPES[1280])}
    assert pose.infer(np.zeros((1080, 1440, 3), dtype=np.uint8), 1280) is None
    assert pose.infer(np.zeros((1080, 1152, 3), dtype=np.uint8), 960) is None
    assert pose.fallback_calls == 2
    assert Fallback.calls == [((1080, 1440, 3), 1280), ((1080, 1152, 3), 960)]
