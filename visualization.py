from autocrop import cropper
from autocrop.utils import resize_image_op
import cv2
import random
import functools


def draw_anchor_bboxes(img, bboxes, shifit=False, color=None, thickness=1):
    change_color = True if color is None else False
    for idx, bbox in enumerate(bboxes):
        font = cv2.FONT_HERSHEY_TRIPLEX
        if shifit:
            dx1, dx2, dy1, dy2 = int((random.random() - 1) * 4), int((random.random() - 1) * 4), \
                                 int((random.random() - 1) * 4), int((random.random() - 1) * 4)
        else:
            dx1, dx2, dy1, dy2 = [0, 0, 0, 0]
        if change_color:
            r, g, b = int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)
            color = (b, g, r)
        if not isinstance(thickness, int):
            thickness = int(thickness)
        cv2.rectangle(img=img,
                      # xmin ymin xmax ymax
                      #    xmin, ymin
                      #   |
                      #   |
                      #   |___________xmax, ymax
                      pt1=tuple([bbox[0] + dx1, bbox[1] + dy1]),
                      pt2=tuple([bbox[2] + dx2, bbox[3] + dy2]),
                      color=color,
                      thickness=thickness)
        cv2.putText(img, str(idx + 1),
                    tuple([bbox[0] + dx1, bbox[1] + dy1]),
                    font, 0.5, (0, 0, 0))


def bbox_visualization(crop_height=1,
                       crop_width=1):
    """
    Visualize anchor bboxes:
    :param rgb_img:
    :param crop_height:
    :param crop_width:
    :return:
    """

    auto_cropper = cropper.AutoCropper(model='mobilenetv2',
                                       cuda=True,
                                       use_face_detector=True)
    img = cv2.imread('imgs/demo.jpg')
    input_img, scale_height, scale_width = resize_image_op(img)
    face_bboxes = auto_cropper.detect_face(input_img)
    raw_face_bboxes = []

    for fbbox in face_bboxes:
        tbbox = [int(round(scale_width * fbbox[0])),
                 int(round(scale_height * fbbox[1])),
                 int(round(scale_width * fbbox[2])),
                 int(round(scale_height * fbbox[3])),
                 ]
        raw_face_bboxes.append(tbbox)
    print(raw_face_bboxes)
    print([(x[2] - x[0]) * (x[3] - x[1]) for x in raw_face_bboxes])
    """
    No filter
    """
    img1 = img.copy()
    generate_bbox_func_partial = functools.partial(auto_cropper.generate_anchor_bboxes,
                                                   image=input_img,
                                                   scale_height=scale_height,
                                                   scale_width=scale_width,
                                                   crop_height=crop_height,
                                                   crop_width=crop_width)
    crop_func_partial = functools.partial(auto_cropper.crop,
                                          rgb_image=img,
                                          topK=1,
                                          crop_width=crop_width,
                                          crop_height=crop_height)
    trans_bboxes, source_bboxes = generate_bbox_func_partial(face_bboxes=[],
                                                             single_face_center=False)
    draw_anchor_bboxes(img=img1, bboxes=raw_face_bboxes, shifit=False, color=(0, 255, 0), thickness=3)

    draw_anchor_bboxes(img=img1, bboxes=source_bboxes, shifit=True)
    cv2.imshow('No Filter, {} Face, {} BBoxes'.format(len(face_bboxes),
                                                      len(source_bboxes)),
               img1)
    cv2.imwrite('No Filter, {} Face, {} BBoxes.jpg'.format(len(face_bboxes),
                                                           len(source_bboxes)),
                img1)
    ret1 = crop_func_partial(filter_face=False,
                             single_face_center=False)
    bbox = ret1[0]
    cv2.imshow("ret1", img[bbox[1]: bbox[3] + 1, bbox[0]: bbox[2] + 1, :])
    cv2.imwrite("ret1.jpg", img[bbox[1]: bbox[3] + 1, bbox[0]: bbox[2] + 1, :])
    """
    Filter Face
    """
    img2 = img.copy()
    trans_bboxes, source_bboxes = generate_bbox_func_partial(face_bboxes=face_bboxes,
                                                             single_face_center=False)

    draw_anchor_bboxes(img=img2, bboxes=raw_face_bboxes, shifit=False, color=(0, 255, 0), thickness=3)

    draw_anchor_bboxes(img=img2, bboxes=source_bboxes, shifit=True)
    cv2.imshow('Filter Face, {} Face, {} BBoxes'.format(len(face_bboxes),
                                                        len(source_bboxes)),
               img2)
    cv2.imwrite('Filter Face, {} Face, {} BBoxes.jpg'.format(len(face_bboxes),
                                                             len(source_bboxes)),
                img2)

    ret2 = crop_func_partial(filter_face=True,
                             single_face_center=False)
    bbox = ret2[0]
    cv2.imshow("ret2", img[bbox[1]: bbox[3] + 1, bbox[0]: bbox[2] + 1, :])
    cv2.imwrite("ret2.jpg", img[bbox[1]: bbox[3] + 1, bbox[0]: bbox[2] + 1, :])

    """
    Filter Center Face
    """
    img3 = img.copy()
    trans_bboxes, source_bboxes = generate_bbox_func_partial(face_bboxes=face_bboxes,
                                                             single_face_center=True)
    draw_anchor_bboxes(img=img3, bboxes=raw_face_bboxes, shifit=False, color=(0, 255, 0), thickness=3)

    draw_anchor_bboxes(img=img3, bboxes=source_bboxes, shifit=True)

    cv2.imshow('Filter Center Face, {} Face, {} BBoxes'.format(len(face_bboxes),
                                                               len(source_bboxes)),
               img3)
    cv2.imwrite('Filter Center Face, {} Face, {} BBoxes.jpg'.format(len(face_bboxes),
                                                                    len(source_bboxes)),
                img3)

    ret3 = crop_func_partial(filter_face=True,
                             single_face_center=True)
    bbox = ret3[0]
    cv2.imshow("ret3", img[bbox[1]: bbox[3] + 1, bbox[0]: bbox[2] + 1, :])
    cv2.imwrite("ret3.jpg", img[bbox[1]: bbox[3] + 1, bbox[0]: bbox[2] + 1, :])

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    bbox_visualization(crop_height=7, crop_width=5)
