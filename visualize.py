import torch
import numpy as np
import cv2

TRESH_HOLD = 0.50

def draw_bbox(img, box, class_name, color=None):
    """
        image: (BGR) numpy array
        box: xywh list
        class_name: string
        color = tupple of 3 values (B,G,R)
    """

    img_h, img_w, channels = img.shape
    x, y, w, h = box
    x, w = x * img_w, w * img_w
    y, h = y * img_h, h * img_h
    xmin = x - w/2
    xmax = x + w/2
    ymin = y - h/2
    ymax = y + h/2
    xmin, xmax, ymin, ymax = [int(i) for i in [xmin, xmax, ymin, ymax]]
    color = (36,255,12)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1)
    cv2.putText(img, class_name, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def visualize_boxes(img, label, name=None, preprocess=None):
    """
    img: torch tensor (BGR)
    label: torch tensor (S, S, B*5 + C)
    """
    # Unormalize
    img = preprocess.unnormalize(img)
    img = preprocess.channel_first_to_channel_last(img)

    # Convert to numpy
    np_img = img.cpu().numpy().astype(np.uint8)
    np_img = preprocess.RGB2BGR(np_img)
    np_labels = label.cpu().numpy()

    for i in range(np_labels.shape[0]):
        for j in range(np_labels.shape[0]):
            bbox1, bbox2 = np_labels[i, j, :5], np_labels[i, j, 5:10]
            for box in [bbox1, bbox2]:
                if box[4] > TRESH_HOLD:
                    class_number = np.argmax(np_labels[i, j, 10:])
                    draw_bbox(np_img, box[:4], str(class_number))

    if not name:
        cv2.imshow("image", np_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(name, np_img)