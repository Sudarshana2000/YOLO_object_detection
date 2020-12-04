import cv2
import numpy as np
import time
import argparse

CONFIDENCE = 0.3
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

config_path = "src/cgf.txt"
weights_path = "src/yolov3.weights"
labels = open("src/images.txt").read().strip().split("\n")
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# generating colors for each object for later plotting
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

# get all the layer names
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def object_detect(image):
    h, w = image.shape[:2]

    # create 4D blob
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

    # sets the blob as the input of the network
    net.setInput(blob)

    # feed forward (inference) and get the network output measure how much it took in seconds
    start = time.perf_counter()
    layer_outputs = net.forward(ln)
    time_took = time.perf_counter() - start
    print(f"Time took: {time_took:.2f}s")

    boxes, confidences, class_ids = [], [], []

    # loop over each of the layer outputs
    for output in layer_outputs:

        # loop over each of the object detections
        for detection in output:

            # extract the class id (label) and confidence (as a probability) of the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # discard weak predictions by ensuring the detected probability is greater than the minimum probability
            if confidence > CONFIDENCE:

                # scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and height
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # perform the non maximum suppression given the scores defined before
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
    font_scale = 1
    thickness = 2

    # ensure at least one detection exists
    if len(idxs) > 0:

        # loop over the indexes we are keeping
        for i in idxs.flatten():

            # extract the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
            text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"

            # calculate text width & height to draw the transparent boxes as background of the text
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
            text_offset_x = x
            text_offset_y = y - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = image.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=2)

            # add opacity (transparency to the box)
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

            # now put the text (label: confidence %)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
    return image


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("input_path", required=True, help="path to input image")
    ap.add_argument("output_path", required=True, help="path to output image")
    ap.add_argument("confidence",type=float, default=0.5,help="minimum probability to filter weak detections")
    ap.add_argument("threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
    args = vars(ap.parse_args())
    global CONFIDENCE,IOU_THRESHOLD
    CONFIDENCE=args.confidence
    IOU_THRESHOLD=args.threshold
    img = cv2.imread(args.input_path)
    output=object_detect(img)
    cv2.imwrite(args.output_path, output)