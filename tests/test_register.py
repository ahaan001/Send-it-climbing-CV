"""Route-image registration: align the video background to a separate route picture."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sendit.register import (align_images, alignment_error, homography_from_corners, plausible,
                             transform_points, transform_pose)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKGROUND = os.path.join(ROOT, "demo_assets", "kilter", "cache", "background.png")


@pytest.fixture(scope="module")
def background():
    img = cv2.imread(BACKGROUND)
    assert img is not None, f"missing demo background: {BACKGROUND}"
    return img


def corners_of(w, h):
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]])


def perturbed_corners(w, h, seed=0, lo=0.04, hi=0.08):
    """Each corner moved by 4-8 % of the image size along each axis, fixed seed."""
    rng = np.random.default_rng(seed)
    sign = rng.choice([-1.0, 1.0], size=(4, 2))
    mag = rng.uniform(lo, hi, size=(4, 2)) * np.array([w, h], np.float32)
    return corners_of(w, h) + sign * mag


@pytest.fixture(scope="module")
def warped_pair(background):
    h, w = background.shape[:2]
    H_true = homography_from_corners(corners_of(w, h), perturbed_corners(w, h, seed=0))
    warped = cv2.warpPerspective(background, H_true, (w, h))
    return H_true, warped


def test_align_images_recovers_perspective_warp(background, warped_pair):
    H_true, warped = warped_pair
    h, w = background.shape[:2]
    H, n = align_images(background, warped)
    assert H is not None
    assert n >= 30
    assert H.shape == (3, 3)
    assert alignment_error(H, H_true, w, h) < 3.0


def test_homography_from_corners_recovers_true_warp(background, warped_pair):
    H_true, _ = warped_pair
    h, w = background.shape[:2]
    src = corners_of(w, h)
    dst = transform_points(H_true, src)
    H = homography_from_corners(src, dst)
    assert alignment_error(H, H_true, w, h) < 1e-3


def test_transform_pose_moves_landmarks_and_keeps_visibility():
    # scale x2 then shift by (+10, -5)
    H = np.array([[2.0, 0.0, 10.0], [0.0, 2.0, -5.0], [0.0, 0.0, 1.0]])
    pose = {"w": 1080, "h": 1920, "fps": 30.0, "n_frames": 3,
            "frames": {0: {"LEFT_WRIST": [100.0, 200.0, 0.93], "NOSE": [50.0, 40.0, 0.5]},
                       2: {"RIGHT_WRIST": [0.0, 0.0, 0.1]}}}
    out = transform_pose(pose, H)
    assert out["frames"][0]["LEFT_WRIST"][:2] == pytest.approx([210.0, 395.0])
    assert out["frames"][0]["LEFT_WRIST"][2] == 0.93
    assert out["frames"][0]["NOSE"][:2] == pytest.approx([110.0, 75.0])
    assert out["frames"][2]["RIGHT_WRIST"] == pytest.approx([10.0, -5.0, 0.1])
    for k in ("w", "h", "fps", "n_frames"):
        assert out[k] == pose[k]
    assert set(out["frames"]) == {0, 2}
    # input untouched
    assert pose["frames"][0]["LEFT_WRIST"] == [100.0, 200.0, 0.93]


def test_align_images_rejects_noise(background):
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 256, size=background.shape, dtype=np.uint8)
    H, n = align_images(background, noise)
    assert H is None
    assert n < 30


def test_plausible_rejects_mirror_and_runaway():
    w, h = 1080, 1920
    assert plausible(np.eye(3), w, h)
    mirror = np.diag([-1.0, 1.0, 1.0])
    assert not plausible(mirror, w, h)
    tiny = np.diag([0.1, 0.1, 1.0])
    assert not plausible(tiny, w, h)
    far = np.array([[1.0, 0.0, 10 * w], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert not plausible(far, w, h)
    assert not plausible(None, w, h)
