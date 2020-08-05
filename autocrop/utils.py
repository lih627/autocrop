import cv2
import numpy as np
from collections import defaultdict


def resize_image_op(raw_img, img_size=256.0):
    """
    :param raw_img: [H, W, 3]
    :param img_size: float: default 256.0 for pretrained model
    :return:
        resized_img: [H', 256, 3] or [256, W', 3]
        scale_height: float H/resized_H
        scale_width: float W/resized_W
    """
    scale = float(img_size) / float(min(raw_img.shape[:2]))
    h = round(raw_img.shape[0] * scale / 32.0) * 32
    w = round(raw_img.shape[1] * scale / 32.0) * 32

    resized_img = cv2.resize(raw_img, (int(w), int(h)))
    scale_height = raw_img.shape[0] / float(resized_img.shape[0])
    scale_width = raw_img.shape[1] / float(resized_img.shape[1])
    return resized_img, scale_height, scale_width


def color_normalization_op(image):
    """
    :param image: [H, W, 3] RGB format
    :return: image: [H, W, 3] color normalization
    """
    RGB_MEAN = (0.485, 0.456, 0.406)
    RGB_STD = (0.229, 0.224, 0.225)
    image = image.astype(np.float32)
    image /= 256.0
    rgb_mean = np.array(RGB_MEAN, dtype=np.float32)
    rgb_std = np.array(RGB_STD, dtype=np.float32)
    image -= rgb_mean
    image /= rgb_std
    return image


def transpose_image_op(image):
    """
    Transpose image as [C, H , W]
    :param image:  [H, W, 3]
    :return:
        [3, H,  W]
    """
    if image.shape[2] == 3:
        image = image.transpose((2, 0, 1))
    return image


def _generate_by_bins(image, n_bins=12):
    """
    :param image:
    :param n_bins:
    :return:
       ---------> w (x)
       |
       |
       |
       | h (y)
       |
       bboxs:  N * 4 [ymin, xmin,,ymax, xmax]
    """
    h = image.shape[0]
    w = image.shape[1]
    step_h = h / float(n_bins)
    step_w = w / float(n_bins)
    annotations = list()
    front = n_bins // 3 + 1
    back = n_bins // 3 * 2
    for x1 in range(0, front):
        for y1 in range(0, front):
            for x2 in range(back, n_bins):
                for y2 in range(back, n_bins):
                    if (x2 - x1) * (y2 - y1) > 0.4 * n_bins * n_bins:
                        annotations.append(
                            [float(step_h * (0.5 + x1)), float(step_w * (0.5 + y1)), float(step_h * (0.5 + x2)),
                             float(step_w * (0.5 + y2))])
    return annotations


def _generate_by_steps(image, height_step, width_step):
    """
    Generate anchor bboxes by steps
    :param image: H W C
    :param height_step: int
    :param width_step:  int
    :return:
       ---------> w (x)
       |
       |
       |
       | h (y)
       |
       bboxs:  N * 4 [ymin, xmin,,ymax, xmax]
    """
    h, w = image.shape[:2]
    h_bins = round(h // height_step)
    w_bins = round(w // width_step)
    annotations = defaultdict(list)
    for h_bin_end in range(h_bins // 3, h_bins + 2):
        for w_bin_end in range(w_bins // 3, w_bins + 2):
            h_end = h_bin_end * height_step
            w_end = w_bin_end * width_step
            if h_end < h and w_end < w:
                for length in range(min(h_bin_end, w_bin_end), 1, -1):
                    h_len = length * height_step
                    w_len = length * width_step
                    h_start = h_end - h_len
                    w_start = w_end - w_len
                    if w_len * h_len > 0.4 * w * h:
                        annotations[0].append([h_start, w_start, h_end, w_end])
                    elif w_len * h_len > 0.3 * w * h:
                        annotations[1].append([h_start, w_start, h_end, w_end])
                    elif w_len * h_len > 0.2 * w * h:
                        annotations[2].append([h_start, w_start, h_end, w_end])
                    elif w_len * h_len > 0.1 * w * h:
                        annotations[3].append([h_start, w_start, h_end, w_end])
    ret = []
    cnt = 0
    for k in [0, 1, 2, 3]:
        if annotations[k]:
            ret.extend(annotations[k])
            cnt += 1
        if cnt == 1:
            break
    return ret


def enlarge_bbox(bbox, factor=1.2):
    """
    Enlarge the bounding boxes
    :param bbox: [N * C] array
    :return:
        [N * 4] xmin ymin xmax ymax
        ---------> w (x)
       |
       |
       |
       | h (y)
       |
    """
    xmin, ymin, xmax, ymax = bbox[:4]
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    w_ = (xmax - xmin) * factor / 2.0
    h_ = (ymax - ymin) * factor / 2.0
    return [cx - w_, cy - h_, cx + w_, cy + h_]


def bbox_intersect_val(bbox1, bbox2):
    """
    1: bbox1 in bbox2
    2: bbox2 in bbox1
    3: IoU(bbox1, bbox2) = 0
    0: interset
    :return:
        int
    """
    x1_min, y1_min, x1_max, y1_max = bbox1[:4]
    x2_min, y2_min, x2_max, y2_max = bbox2[:4]
    if x1_min >= x2_min and y1_min >= y2_min and x1_max <= x2_max and y1_max <= y2_max:
        return 1
    if x2_min >= x1_min and y2_min >= y1_min and x2_max <= x1_max and y2_max <= y1_max:
        return 2
    if x1_max <= x2_min or x2_max <= x1_min or y1_max <= y2_min or y2_max <= y1_min:
        return 3
    return 0


def _filter_single_face_anchor_bbox(face_bbox, anchor_bbox):
    """
    ---------> w (x)
    |
    |
    |
    | h (y)
    |
    Return True if face_bbox in the x-axis center of anchor_bbox
    :param face_bbox: [xmin ymin xmax ymax]
    :param anchor_bbox: [xmin ymin xmax ymax]
    :return: bool
    """
    if isinstance(face_bbox[0], list):
        face_bbox = face_bbox[0]
    anchor_width_center = (anchor_bbox[0] + anchor_bbox[2]) / 2.0
    left_length = anchor_width_center - face_bbox[0]
    right_length = face_bbox[2] - anchor_width_center
    # print("f {}, a {}, center {},\n left {}, right {}, ratio {}".format(face_bbox,
    #                                                                     anchor_bbox,
    #                                                                     anchor_width_center,
    #                                                                     left_length,
    #                                                                     right_length,
    #                                                                     abs(right_length - left_length) / (
    #                                                                             right_length + left_length)
    #                                                                     ))
    if left_length < 0:
        return False
    if right_length < 0:
        return False
    return abs(right_length - left_length) / (right_length + left_length) < 0.5


def _filter_double_face_anchor_bbox(face_bbox, anchor_bbox):
    """
    ---------> w (x)
    |
    |
    |
    | h (y)
    |
    Return True if face_bbox in the x-axis center of anchor_bbox
    :param face_bboxes: N * 4, N * [xmin ymin xmax ymax]
    :param anchor_bbox: [xmin ymin xmax ymax]
    :return: bool
    """
    anchor_width_center = (anchor_bbox[0] + anchor_bbox[2]) / 2.0
    ctr = []
    for fbbox in face_bbox:
        ctr.append((fbbox[0] + fbbox[2]) / 2.0)
    l1 = anchor_width_center - ctr[0]
    l2 = anchor_width_center - ctr[1]
    # print(l1, l2)
    return -1.1 < l1 / l2 < -0.9


def generate_bboxes(resized_image,
                    scale_height,
                    scale_width,
                    crop_height=None,
                    crop_width=None,
                    min_step=4,
                    n_bins=14,
                    face_bboxes=None,
                    single_face_center=True,
                    double_face_center=True):
    """
    generate transformed_bbox and source_bbox.
    Use transformed_bboxes as rois for post-processing.
    note:
        The coordinate definition order of the trans_bboxes
         and source_bboxes
         is different

    :param resized_image: resized_img H W 3, one of [H, W] is 255
    :param scale_height: original image height / resized image height
    :param scale_width: original image width / resized image width
    :param crop_height:
    :param crop_width:
    :param min_step: the minimum step_size: defalut 8
    :param single_face_center: bool True: face bbox will in the center of the anchor bboxes
    :return:
        ---------> w (x)
        |
        |
        |
        | h (y)
        |
        trans_bboxes  [Nx4] [xmin ymin xmax ymax]
        source_bboxes [Nx4] [xmin ymin xmax ymax]
    """
    if crop_height is None or crop_width is None:
        bboxes = _generate_by_bins(resized_image, n_bins=n_bins)
    else:
        h_w_ratio = crop_height / float(crop_width)
        if h_w_ratio < 1.0:
            h_step = min_step
            w_step = round(crop_width * min_step / float(crop_height))
        else:
            w_step = min_step
            h_step = round(crop_height * min_step) / float(crop_width)
        bboxes = _generate_by_steps(resized_image, height_step=h_step, width_step=w_step)
    # bboxes ymin xim ymax xmax
    source_bboxes = []
    trans_bboxes = []
    for bbox in bboxes:
        # trans_bboxes xmin ymin xmax ymax
        trans_bboxes.append([round(item) for item in [bbox[1], bbox[0],
                                                      bbox[3], bbox[2]]])
        # source_bboxes ymin xmin ymax xmax
        source_bboxes.append([round(item) for item in [bbox[1] * scale_width,
                                                       bbox[0] * scale_height,
                                                       bbox[3] * scale_width,
                                                       bbox[2] * scale_height]])

    if face_bboxes is not None:
        """
        Filter out small faces
        """
        if len(face_bboxes) > 0:
            area = []
            for item in face_bboxes:
                area.append((item[2] - item[0]) * (item[3] - item[1]))
            max_area = max(area)
            filter_face_bboxes = []
            for idx, face_bbox in enumerate(face_bboxes):
                if area[idx] / max_area > 0.5:
                    filter_face_bboxes.append(face_bbox)
            ret = {'t_bboxes': [],
                   's_bboxes': [],
                   'num_face': [],
                   'face_bboxes': []}

            for idx, tbbox in enumerate(trans_bboxes):
                is_valied = True
                can_used = False
                num_face = 0
                cur_face_bboxes = []
                for fbbox in filter_face_bboxes:
                    i_val = bbox_intersect_val(tbbox, fbbox)
                    if i_val == 0:
                        is_valied = False
                        break
                    elif i_val == 2:  # fbbox in tbbox
                        cur_face_bboxes.append(fbbox)
                        num_face += 1
                        can_used = True
                if is_valied and can_used:
                    ret['t_bboxes'].append(tbbox)
                    ret['s_bboxes'].append(source_bboxes[idx])
                    ret['num_face'].append(num_face)
                    ret['face_bboxes'].append(cur_face_bboxes)
            if single_face_center or double_face_center:
                t_bboxes = []
                s_bboxes = []

                for idx, num_face in enumerate(ret['num_face']):
                    if num_face not in [1, 2]:
                        t_bboxes.append(ret['t_bboxes'][idx])
                        s_bboxes.append(ret['s_bboxes'][idx])
                    elif num_face == 1 and not single_face_center or _filter_single_face_anchor_bbox(
                            face_bbox=ret['face_bboxes'][idx],
                            anchor_bbox=ret['t_bboxes'][idx]):
                        t_bboxes.append(ret['t_bboxes'][idx])
                        s_bboxes.append(ret['s_bboxes'][idx])
                    elif num_face == 2 and (not double_face_center or _filter_double_face_anchor_bbox(
                            face_bbox=ret['face_bboxes'][idx],
                            anchor_bbox=ret['t_bboxes'][idx]
                    )):
                        t_bboxes.append(ret['t_bboxes'][idx])
                        s_bboxes.append(ret['s_bboxes'][idx])
                    # if num_face != 1 or _filter_single_face_anchor_bbox(
                    #         face_bbox=ret['face_bboxes'][idx],
                    #         anchor_bbox=ret['t_bboxes'][idx]):
                    #     t_bboxes.append(ret['t_bboxes'][idx])
                    #     s_bboxes.append(ret['s_bboxes'][idx])
                # print(len(t_bboxes))
                if len(t_bboxes) > 1:
                    return t_bboxes, s_bboxes

            if len(ret['t_bboxes']) > 1:
                return ret['t_bboxes'], ret['s_bboxes']

    return trans_bboxes, source_bboxes


if __name__ == '__main__':
    pass
