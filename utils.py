from copy import copy
from typing import Tuple

import cv2
import numpy as np


class BoundingBox:
    def __init__(
        self, x: int, y: int, w: int, h: int, object_x: int = None, object_y: int = None
    ):
        self.x = x
        self.y = y
        self.w = w if w % 2 == 0 else w + 1
        self.h = h if h % 2 == 0 else h + 1
        self.object_x = object_x if object_x is not None else x
        self.object_y = object_y if object_y is not None else y

    def clip(
        self, height: int, width: int, target_bbox_h: int, target_bbox_w: int
    ) -> bool:
        """
        Clip the bounding box to the frame dimensions. Returns True if the bounding
        box was clipped, False otherwise.
        """

        x = max(0, min(self.x, width))
        y = max(0, min(self.y, height))

        w, h = target_bbox_w, target_bbox_h
        if self.x < target_bbox_w // 2:
            w = 2 * self.x
        if self.y < target_bbox_h // 2:
            h = 2 * self.y

        if target_bbox_w > 2 * (width - self.x):
            w = 2 * (width - self.x)
        if target_bbox_h > 2 * (height - self.y):
            h = 2 * (height - self.y)

        clipped = x != self.x or y != self.y or w != self.w or h != self.h
        self.x, self.y, self.w, self.h = x, y, w, h
        return clipped

    @property
    def left(self) -> int:
        return self.x - self.w // 2

    @property
    def top(self) -> int:
        return self.y - self.h // 2

    @property
    def right(self) -> int:
        return self.x + self.w // 2

    @property
    def bottom(self) -> int:
        return self.y + self.h // 2

    def __repr__(self) -> str:
        return (
            f"BoundingBox(x={self.x}, y={self.y}, w={self.w}, h={self.h}, "
            f"object_x={self.object_x}, object_y={self.object_y}, "
            f"left={self.left}, top={self.top}, right={self.right}, bottom={self.bottom})"
        )


class MosseResult:
    def __init__(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        target_response: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        f: np.ndarray,
    ):
        self.frame = frame
        self.bbox = bbox
        self.target_response = target_response

        self.filter = A / B
        G = self.filter * np.fft.fft2(f)
        self.filter = np.abs(normalize_range(np.fft.ifft2(self.filter)))
        self.output = np.abs(normalize_range(np.fft.ifft2(G)))


def gauss_reponse(
    height: int,
    width: int,
    center_x: int,
    center_y: int,
    sigma: float = 10.0,
    normalize: bool = True,
) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    response = np.exp(-((xs - center_x) ** 2 + (ys - center_y) ** 2) / (2 * sigma**2))
    if normalize:
        response = normalize_range(response)
    return response


def normalize_range(x: np.ndarray) -> np.ndarray:
    """Normalize the input to have a minimum of 0 and a maximum of 1."""
    return (x - x.min()) / (x.max() - x.min())


def log_transform(image: np.ndarray) -> np.ndarray:
    return np.log(image + 1)


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalize the image to have a mean of 0 and a standard deviation of 1."""
    return (image - image.mean()) / (image.std() + 1e-5)


def hanning_window(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    mask_col, mask_row = np.meshgrid(np.hanning(width), np.hanning(height))
    window = mask_col * mask_row
    return image * window


def preprocess(image: np.ndarray) -> np.ndarray:
    image = log_transform(image)
    image = normalize(image)
    image = hanning_window(image)
    return image


def random_affine_transform(
    image: np.ndarray,
    bbox: BoundingBox,
    rotation: float = 180 / 16,
    scale: float = 0.05,
    translation: int = 4,
) -> Tuple[np.ndarray, BoundingBox]:
    new_bbox = copy(bbox)

    angle = np.random.uniform(-rotation, rotation)
    matrix = cv2.getRotationMatrix2D((new_bbox.x, new_bbox.y), angle, 1.0)
    image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

    scale = np.random.uniform(1 - scale, 1 + scale)
    image = cv2.resize(image, None, fx=scale, fy=scale)

    trans_x = np.random.randint(-translation, translation)
    trans_y = np.random.randint(-translation, translation)
    new_bbox.object_x += trans_x
    new_bbox.object_y += trans_y

    return image, new_bbox
