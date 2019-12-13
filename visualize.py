import torch
import numpy as np
import cv2

from voc_classes import get_class_name

def draw_bbox(img, box, text, color, thickness=2):
    """
        image: (BGR) numpy array
        box: list of [xmin, ymin, xmax, ymax]
        class_name: string
        color = tupple of 3 values (B,G,R)
    """

    xmin, ymin, xmax, ymax = [int(i) for i in box]
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
    cv2.putText(img, text, (xmin, max(ymin-10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, thickness=1)


def draw_all_bboxes(np_img, bboxes, preprocess, color, name=None):
    """
    np_img: opencv numpy array (BGR)
    bboxes: list of bounding boxes - box = [x1,y1,x2,y2,class,confidence]
    color: tuble (B, G, R)
    """

    for bbox in bboxes:
        class_number = int(bbox[4])
        confidence = round(bbox[5] * 100)
        box_label = "{} {}%".format(
                    get_class_name(class_number), confidence)
        draw_bbox(np_img, bbox[:4], box_label, color, thickness=2)

    if not name:
        cv2.imshow("image", np_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(name, np_img)


def print_cell_with_objects(label):
    np_labels = label.cpu().numpy()
    for i in range(np_labels.shape[0]):
        for j in range(np_labels.shape[0]):
            bbox1, bbox2 = np_labels[i, j, :5], np_labels[i, j, 5:10]
            for box in [bbox1, bbox2]:
                if box[4] > TRESH_HOLD:
                    class_number = np.argmax(np_labels[i, j, 10:])
                    print("object in cell {}-{}: {}", j, i, box)
