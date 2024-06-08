from typing import List, Tuple

import cv2
import numpy as np

from utils import (
    BoundingBox,
    MosseResult,
    gauss_reponse,
    normalize_range,
    preprocess,
    random_affine_transform,
)


class Mosse:
    def __init__(self, sigma: float = 10, learning_rate: float = 0.125):
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.first_frame = True
        self.clipped = False

    def init(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        pretrain_iters: int = 128,
    ) -> List[MosseResult]:
        """Pretrain the filter on the first frame."""

        results = []

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.bbox = bbox

        response = gauss_reponse(*frame.shape, bbox.x, bbox.y)

        g = response[self.bbox.top : self.bbox.bottom, self.bbox.left : self.bbox.right]
        f = frame[self.bbox.top : self.bbox.bottom, self.bbox.left : self.bbox.right]
        G = np.fft.fft2(g)

        f = preprocess(f)
        self.A_i = G * np.conj(np.fft.fft2(f))
        self.B_i = np.fft.fft2(f) * np.conj(np.fft.fft2(f))

        results.append(MosseResult(frame, self.bbox, response, self.A_i, self.B_i, f))

        for _ in range(pretrain_iters):
            trans_frame, bbox = random_affine_transform(frame, self.bbox)
            f = trans_frame[bbox.top : bbox.bottom, bbox.left : bbox.right]
            response = gauss_reponse(*trans_frame.shape, bbox.object_x, bbox.object_y)

            g = response[bbox.top : bbox.bottom, bbox.left : bbox.right]
            G = np.fft.fft2(g)
            f = preprocess(f)

            self.A_i += G * np.conj(np.fft.fft2(f))
            self.B_i += np.fft.fft2(f) * np.conj(np.fft.fft2(f))

            results.append(
                MosseResult(trans_frame, self.bbox, response, self.A_i, self.B_i, f)
            )

        self.A_i *= self.learning_rate
        self.B_i *= self.learning_rate
        self.H_i = self.A_i / self.B_i

        self.search_win_h = self.H_i.shape[0]
        self.search_win_w = self.H_i.shape[1]

        results.append(MosseResult(frame, self.bbox, response, self.A_i, self.B_i, f))
        return results

    def update(self, frame: np.ndarray) -> BoundingBox:
        """Update the tracker with the new frame."""

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        f = frame[self.bbox.top : self.bbox.bottom, self.bbox.left : self.bbox.right]
        f = preprocess(f)

        if f.shape != (self.search_win_h, self.search_win_w):
            f = cv2.resize(f, (self.search_win_w, self.search_win_h))
            self.clipped = False

        G = self.H_i * np.fft.fft2(f)
        g = normalize_range(np.fft.ifft2(G))

        max_pos = np.where(g == g.max())
        self.bbox.x = max_pos[1].item() + self.bbox.left
        self.bbox.y = max_pos[0].item() + self.bbox.top

        # TODO: Make sure the bounding box doesn't go out of bounds

        self.clipped = self.bbox.clip(
            *frame.shape, self.search_win_h, self.search_win_w
        )

        f = frame[self.bbox.top : self.bbox.bottom, self.bbox.left : self.bbox.right]
        f = preprocess(f)

        if f.shape != (self.search_win_h, self.search_win_w):
            f = cv2.resize(f, (self.search_win_w, self.search_win_h))
        G = self.H_i * np.fft.fft2(f)

        self.A_i = (
            self.learning_rate * (G * np.conj(np.fft.fft2(f)))
            + (1 - self.learning_rate) * self.A_i
        )
        self.B_i = (
            self.learning_rate * (np.fft.fft2(f) * np.conj(np.fft.fft2(f)))
            + (1 - self.learning_rate) * self.B_i
        )

        self.H_i = self.A_i / self.B_i

        return self.bbox
