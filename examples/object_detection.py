#!/usr/bin/env python3
"""Object detection example using BracketBot AI"""

# /// script
# dependencies = [
#   "bracketbot-ai @ /home/bracketbot/bracketbot-ai/dist/bracketbot_ai-0.0.1-py3-none-any.whl",
#   "opencv-python",
# ]
# ///

import subprocess
from bracketbot_ai import Detector
import cv2

# Download a sample image
subprocess.run(["wget", "https://picsum.photos/id/88/200/300", "-O", "test.jpg"])

detector = Detector("yolo11s")
img = cv2.imread("test.jpg")
results = detector(img)
print("Number of detections: ", len(results))
cv2.imwrite("test_result.jpg", results.plot())