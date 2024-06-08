import argparse
import time

import cv2

from mosse import Mosse, BoundingBox

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str)
    args = parser.parse_args()

    tracker = Mosse()
    cap = cv2.VideoCapture(args.video)

    ret, frame = cap.read()
    bbox = cv2.selectROI("Select the object to track", frame)

    bbox = BoundingBox(bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2, bbox[2], bbox[3])
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bbox = tracker.update(frame)
        cv2.rectangle(
            frame,
            (bbox.left, bbox.top),
            (bbox.right, bbox.bottom),
            color=(0, 255, 0),
            thickness=2,
        )
        cv2.imshow("Tracking", frame)

        time.sleep(0.025)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
