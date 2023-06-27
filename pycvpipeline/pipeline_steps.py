from typing import Any, Optional
from uuid import uuid4

try:
    import cv2 as cv
except ImportError:
    from cv2 import cv2 as cv

import numpy as np

from .utils import TContours, is_gray, is_rgb, write_svg


def set_config(cfg: Optional[dict[str, Any]] = None):
    config = {
        "canny_thr1": 10,
        "canny_thr2": 200,
        "gauss_ker": (3, 3),
        "gauss_sigma": 0,
        "contour_mode": cv.RETR_TREE,
        "contour_method": cv.CHAIN_APPROX_SIMPLE,
        "contour_idx": -1,
        "contour_rgb": (255, 20, 147),  # pink
        "contour_thickess": 2,
        "rgb_before_contour": True,
        "save_contours": True,
        "save_contours_path": "contours_out/",
        "morph_dilate_kernel": cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 1)),
        "morph_close_kernel": cv.getStructuringElement(cv.MORPH_RECT, (3, 3)),
        "morph_dilate_iters": 2,
        "morph_close_iters": 2,
        "morph_close_border": cv.BORDER_REFLECT101,
    }
    if cfg:
        config.update(cfg)
    return config


CONFIG = set_config()


def to_gray(img: np.ndarray) -> np.ndarray:
    assert is_rgb(
        img
    ), f"Image must be RGB to convert to grayscale, image has dim={len(img.shape)}"
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def to_rgb(img: np.ndarray) -> np.ndarray:
    assert is_gray(
        img
    ), f"Image must be grayscale to convert to RGB, image has dim={len(img.shape)}"
    return cv.cvtColor(img, cv.COLOR_GRAY2RGB)


def gauss_blur(img: np.ndarray) -> np.ndarray:
    return cv.GaussianBlur(img, CONFIG["gauss_ker"], CONFIG["gauss_sigma"])


def canny(img: np.ndarray) -> np.ndarray:
    # add --generate-members to pylint args for cv.Canny to not show missing error
    return cv.Canny(
        img, threshold1=CONFIG["canny_thr1"], threshold2=CONFIG["canny_thr1"]
    )


def dilate_morphology(img: np.ndarray) -> np.ndarray:
    return cv.dilate(
        img,
        kernel=CONFIG["morph_dilate_kernel"],
        iterations=CONFIG["morph_dilate_iters"],
    )


def close_morphology(img: np.ndarray) -> np.ndarray:
    """Closes lines that have small holes"""
    return cv.morphologyEx(
        img,
        kernel=CONFIG["morph_close_kernel"],
        op=cv.MORPH_CLOSE,
        iterations=CONFIG["morph_close_iters"],
        borderType=CONFIG["morph_close_border"],
    )


def contour(img: np.ndarray) -> np.ndarray:
    contours: TContours
    print("Sub-step find_contours running...")
    contours, _ = cv.findContours(img, CONFIG["contour_mode"], CONFIG["contour_method"])
    assert len(contours) > 0, f"No countours found: {contours}"
    print("Sub-step find_contours finish.")

    if CONFIG["rgb_before_contour"]:
        print("Sub-step to_rgb running...")
        img = to_rgb(img.copy())
        print("Sub-step to_rgb finish.")
    if CONFIG["save_contours"]:
        print("Sub-step write_svg running...")
        _id = str(uuid4())
        print(f"Image id={_id}")
        write_svg(
            img,
            contours,
            CONFIG["save_contours_path"] + "svg_contours" + "_" + _id + ".svg",
        )
        print("Sub-step write_svg finish.")

    cv.drawContours(
        img,
        contours,
        CONFIG["contour_idx"],
        CONFIG["contour_rgb"],
        CONFIG["contour_thickess"],
    )
    return img
