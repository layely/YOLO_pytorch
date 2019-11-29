import torch

def IOU(x1, y1, w1, h1, x2, y2, w2, h2):
    return 0

def mAP():
    return 0

def confidence(x, y, w, h, ground_truths):
    """
        Given a bbox defined by x,y,w,h and the coordonates
        of all objects in an image. Compute the confidence.
        ground_truths should be a 'list' of bboxes (I mean a tensor).
        return the max IOU. If there is no object, then confidence should
        be 0.
    """
    max_iou = 0

    for i in ground_truths.shape[0]:
        actual_bbox = ground_truths[i]
        gt_x = actual_bbox[0].item()
        gt_y = actual_bbox[1].item()
        gt_w = actual_bbox[2].item()
        gt_h = actual_bbox[3].item()
        iou = IOU(x, y, w, h, gt_x, gt_y, gt_w, gt_h)
        max_iou = max(max_iou, iou)

    return max_iou



