import object_detection as od
import cv2
import time
import argparse

def video_object_detect(video,path):
    cap=cv2.VideoCapture(video)
    res,image=cap.read()
    h,w=image.shape[:2]
    fourcc=cv2.VideoWriter_fourcc(*"XVID")
    out=cv2.VideoWriter(path,fourcc,20.0,(w,h))
    while True:
        res,image=cap.read()
        if image is None:
            break
        img=od.object_detect(image)
        out.write(img)
    cap.release()
    out.release()


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("input_path", required=True, help="path to input video")
    ap.add_argument("output_path", required=True, help="path to output video")
    ap.add_argument("confidence",type=float, default=0.5,help="minimum probability to filter weak detections")
    ap.add_argument("threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
    args = vars(ap.parse_args())
    od.CONFIDENCE=args.confidence
    od.IOU_THRESHOLD=args.threshold
    video_object_detect(args.input_path,args.output_path)
