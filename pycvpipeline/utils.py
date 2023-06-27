from pathlib import Path
from random import random

import numpy as np


def is_gray(img: np.ndarray):
    return len(img.shape) == 2  # 2D shape means only 1 channel, 3D shape means RGB


def is_rgb(img: np.ndarray):
    return len(img.shape) == 3


TContours = list[list[list[list[int]]]]


def get_random_rgba(opacity=1):
    return [random() * 255 for _ in range(3)] + [opacity]


SVG_STYLE = "stroke:pink;stroke-width:2px;pointer-events:all"


def random_fill(style: str):
    return style + ";fill:rgba(" + ", ".join(str(x) for x in get_random_rgba()) + ")"


def write_svg(img: np.ndarray, contours: TContours, outname: str):
    if is_gray(img):
        height, width = img.shape
    else:
        height, width, _ = img.shape
    with Path(f"{outname}").open("w+", encoding="utf-8") as outf:
        outf.write(
            f'<svg width="{width}" height="{height}"'
            ' xmlns="http://www.w3.org/2000/svg">'
        )

        for c in contours:
            outf.write('<path d="M')
            for v in c:
                x, y = v[0]
                outf.write(f"{x} {y} ")
            outf.write(
                f'" style="{random_fill(SVG_STYLE)}" '  # noqa
                f' id="contour{c[0][0][0]}_{c[0][0][1]}"/>'
            )
        outf.write("</svg>")
