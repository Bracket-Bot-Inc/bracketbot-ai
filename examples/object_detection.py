# /// script
# dependencies = [
#   "bracketbot-ai",
#   "opencv-python",
# ]
# [tool.uv.sources]
# bracketbot-ai = { path = "/home/bracketbot/bracketbot-ai", editable = true }
# ///

import subprocess
from bracketbot_ai import Detector
import cv2

detector = Detector("yolo11s", device=0)
results = detector(img, conf=0.35, iou=0.45)
print("Number of detections: ", len(results))
cv2.imwrite("test_result.jpg", results.plot())

if __name__ == "__main__":
    sys.exit(main()) 