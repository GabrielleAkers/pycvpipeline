from pathlib import Path
from typing import Callable, Generator, Self, TypeVar

try:
    import cv2 as cv
except ImportError:
    from cv2 import cv2 as cv

import numpy as np
from matplotlib import pyplot as plt

from .utils import TContours

_T = TypeVar("_T", np.ndarray, TContours)

TStep = Callable[[_T], _T]


class CvPipeline:
    _name: str
    _orig: np.ndarray
    _op: np.ndarray
    _steps: list[TStep]
    __steps_gen: Generator

    def __init__(self, name: str, img_path: Path | str, steps: list[TStep]) -> None:
        self._name = name
        self._orig = self._try_load_img(img_path)
        self._op = np.copy(self._orig)
        self._steps = steps
        self.__steps_gen = self._build_gen(steps)

    def _try_load_img(self, path: Path | str):
        assert Path(path).exists(), f"Path {path} does not exist"
        img = cv.imread(str(path), cv.COLOR_GRAY2RGBA)
        assert img is not None
        return img

    def _build_gen(self, steps: list[TStep]):
        for step in steps:
            yield step

    def run(
        self,
        save_intermediate=False,
        base_path: Path | str = "",
        outname: str = "",
    ) -> Self:
        for step in self.__steps_gen:
            self._op = step(self._op)
            if save_intermediate:
                self.save(base_path, outname + "_" + step.__name__ + ".png")
        return self

    def next_step(self, num=1) -> Self:
        try:
            for _ in range(num):
                s = next(self.__steps_gen)
                print(f"Step {s.__name__} running...")
                self._op = s(self._op)
                print(f"Step {s.__name__} finish.\n")
        except StopIteration:
            print("No more steps.")
        return self

    def show_result_img(self) -> Self:
        if not isinstance(self._op, np.ndarray):
            print(f"Unable to display type: {type(self._op)}")
            return self
        print("Result:")
        _ = plt.subplot(121), plt.imshow(self._orig, cmap="gray")

        _ = plt.title("Original Image"), plt.xticks([]), plt.yticks([])
        _ = plt.subplot(122), plt.imshow(self._op, cmap="gray")
        _ = plt.title("Result Image"), plt.xticks([]), plt.yticks([])
        plt.show()
        return self

    def save(self, base_path: Path | str, outpath: Path | str) -> Self:
        out = str(Path(base_path) / (self._name + str(outpath)))
        print(f"Saving to {out}...")
        cv.imwrite(out, self._op)
        return self

    def __repr__(self):
        return (
            "Pipeline(steps="
            f"{[str(i) + ' : '  + f.__name__ for (i, f) in enumerate(self._steps)]}"
        )
