import torch
import numpy as np
import cv2

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

def draw_bbox(img, box, text, color=None, thickness=2):
    """
        image: (BGR) numpy array
        box: list of [xmin, ymin, xmax, ymax]
        class_name: string
        color = tupple of 3 values (B,G,R)
    """

    xmin, ymin, xmax, ymax = [int(i) for i in box]
    color = (0, 255, 0)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
    cv2.putText(img, text, (xmin, max(ymin-10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, thickness=1)


def visualize_boxes(img, label, name=None, preprocess=None):
    """
    img: torch tensor (BGR)
    label: torch tensor (S, S, B*5 + C)
    """

    # Get numpy image in opencv format
    np_img = preprocess.post_process_image(img)
    # get bounding boxes + class + confidence
    img_h, img_w, _ = np_img.shape
    bboxes = preprocess.decode_label(label, img_h, img_w)

    for bbox in bboxes:
        class_number = int(bbox[4])
        confidence = round(bbox[5] * 100)
        box_label = "{} {}%".format(
                    VOC_CLASSES[class_number], confidence)
        draw_bbox(np_img, bbox[:4], box_label, thickness=2)

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
