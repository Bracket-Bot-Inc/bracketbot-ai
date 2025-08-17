# /// script
# dependencies = [
#   "bracketbot-ai",
#   "opencv-python",
#   "mjpeg-streamer"
# ]
# [tool.uv.sources]
# bracketbot-ai = { path = "/home/bracketbot/bracketbot-ai", editable = true }
# ///

import argparse
import os
import subprocess
import time
from pathlib import Path
from mjpeg_streamer import MjpegServer, Stream
from bracketbot_ai import Detector
import cv2

def main():
    parser = argparse.ArgumentParser(description="BracketBot AI Object Detector")
    parser.add_argument('model', help='Path to .rknn model file (checks models/ dir)')
    parser.add_argument('source', nargs='?', help='Image, video path, or camera index')
    parser.add_argument('--stream', action='store_true', help='Stream results')
    parser.add_argument('--save', action='store_true', help='Save results')
    parser.add_argument('--conf', type=float, default=0.35, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference size')
    parser.add_argument('--device', type=int, default=0, choices=[-1,0,1,2], help='NPU device')
    parser.add_argument('--info', action='store_true', help='Show model info')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        model = Detector(args.model, device=args.device, verbose=args.verbose)

        if args.source is None:
            # Download a sample image
            if not os.path.exists("test.jpg"):
                subprocess.run(["wget", "https://picsum.photos/id/88/200/300", "-O", "test.jpg"])
            img = cv2.imread("test.jpg")
        if args.info:
            model.info()
            return 0
        print(f"\nProcessing: {args.source}")
        print(f"Settings: conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}")
        
        # Handle video/camera streams
        if args.source.isdigit() or Path(args.source).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
            print("Streaming mode (press 'q' to quit)...\n")

            # Open video source
            cap_source = int(args.source) if args.source.isdigit() else args.source
            cap = cv2.VideoCapture(cap_source, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {args.source}")

            ret = False
            while not ret:
                ret, frame = cap.read()
            if args.stream:
                stream = Stream("my_camera", size=(frame.shape[0], frame.shape[1]), quality=50, fps=30)
                server = MjpegServer("localhost", 8080)
                server.add_stream(stream)
                server.start()
            frame_count = total_detections = 0
            save_dir = Path("runs/detect") if args.save else None
            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)
            speeds_avg = {}
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Run detection on single frame
                    results = model(frame, conf=args.conf, iou=args.iou)
                    for speed in results.speed:
                        if speed not in speeds_avg:
                            speeds_avg[speed] = 0
                        speeds_avg[speed] += results.speed[speed]
                    frame_count += 1
                    total_detections += len(results.boxes)
                    print(f"\rFrame {frame_count}: {len(results.boxes)} objects", end='', flush=True)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    cv2.imwrite("preprocess.jpg", results.orig_img)
                    # Save frame
                    if args.save:
                        save_path = save_dir / f"frame_{frame_count:06d}.jpg"
                        save_img = results.plot()
                        if save_img is not None:
                            cv2.imwrite(str(save_path), save_img)
                    if args.stream:
                        # Stream results to localhost:PORT
                        stream.set_frame(results.plot())
                    time.sleep(0.05)
            except KeyboardInterrupt:
                print("\n\nStopped by user")
            finally:
                print(f"\n\nProcessed {frame_count} frames, {total_detections} total detections")
                for speed in speeds_avg:
                    print(f"{speed}: {speeds_avg[speed]/frame_count:.1f}ms")
                if frame_count > 0: 
                    print(f"Average: {total_detections/frame_count:.1f} objects/frame")
                    if args.save:
                        print(f"Saved frames to: {save_dir}")
                if args.stream:
                    server.stop()
                cap.release()
        else:
            # Handle single image
            results = model(args.source, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                          save=args.save, show=args.show)
            
            print(f"\nResults: {len(results)} objects detected")
            
            if args.verbose and len(results) > 0:
                print("\nDetections:")
                for i, box in enumerate(results.boxes):
                    x1, y1, x2, y2, conf, cls = box
                    print(f"  [{i}] {results.names[int(cls)]}: {conf:.3f} at ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
                print(f"\nSpeed: {sum(results.speed.values()):.1f}ms total")
            
            if args.save:
                print(f"\nSaved to: {results.save()}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    main()