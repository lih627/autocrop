import torch
import face_detection
from torch.autograd import Variable
from .model.cropping_model import build_crop_model
from .utils import resize_image_op, color_normalization_op, enlarge_bbox, generate_bboxes, transpose_image_op


class AutoCropper(object):
    def __init__(self,
                 model: str = 'mobilenetv2',
                 cuda: bool = True,
                 model_path: str = "",
                 use_face_detector: bool = True):
        """
        Generate an autocropper instance
        :param model: str mobilenetv2 or shufflenetv2
        :param cuda: bool set True if cuda if inference on GPU
        :param model_path: str
        :param use_face_detector: bool: default True
        """
        self.cuda = True if cuda and torch.cuda.is_available() else False
        self.cropper = build_crop_model(model=model, cuda=self.cuda, model_path=model_path)
        self.face_detector = None
        if use_face_detector:
            self.face_detector = face_detection.build_detector("DSFDDetector",
                                                               confidence_threshold=0.5,
                                                               nms_iou_threshold=0.3)
        if self.cuda:
            self.cropper = torch.nn.DataParallel(self.cropper)
            self.cropper.cuda()

    def detect_face(self, rgb_img):
        """
        Detect face bbox from rgb_img and enlarge the bbox
        See utils.
        :param rgb_img:
        :return:
            bboxes List[np.array[4]] [xmin ymin xmax ymax]
        """
        return [enlarge_bbox(bbox) for bbox in self.face_detector.detect(rgb_img)]

    def eval_rois(self, image, roi):
        """
        :param image: H W 3 RGB format
        :param roi: []
        :return:
            roi index
        """
        image = transpose_image_op(image)
        image = torch.unsqueeze(torch.as_tensor(image), 0)
        if self.cuda:
            resized_rgb_img_torch = Variable(image.cuda())
            roi = Variable(torch.Tensor(roi))
        else:
            resized_rgb_img_torch = Variable(image)
            roi = Variable(torch.Tensor(roi))
        out = self.cropper(resized_rgb_img_torch, roi)
        id_out = sorted(range(len(out)), key=lambda k: out[k], reverse=True)
        return id_out

    def generate_anchor_bbxoes(self,
                               image,
                               scale_width,
                               scale_height,
                               crop_width,
                               crop_height,
                               face_bboxes):
        """
        See autocrop.utils.generate_bboxes for details
        """
        return generate_bboxes(image,
                               scale_width=scale_width,
                               scale_height=scale_height,
                               crop_width=crop_width,
                               crop_height=crop_height,
                               face_bboxes=face_bboxes)

    def crop(self,
             rgb_image,
             topK=1,
             crop_height=None,
             crop_width=None,
             filter_face=True):
        """
        Crop the image and return TopK crop results' coordinate: List[list]
        coordinate for each bbox is defined as [xmin ymin xmax ymax]
        :param rgb_image: [H W 3] RGB format
        :param topK: int: default return Top 1 cropped result
        :param crop_height: height ratio
        :param crop_width: width ratio
        :param filter_face: use face detection to filter roi
        :return:
            cropped bboxes:
            List[list[4]] list[4] : [xmin ymin xmax ymax]
        """
        input_img, scale_height, scale_width = resize_image_op(rgb_image)  # H W 3 RGB format
        face_bboxes = []
        if self.face_detector and filter_face:
            face_bboxes = self.detect_face(input_img)

        trans_bboxes, source_bboxes = self.generate_anchor_bbxoes(input_img,
                                                                  scale_width=scale_width,
                                                                  scale_height=scale_height,
                                                                  crop_height=crop_height,
                                                                  crop_width=crop_width,
                                                                  face_bboxes=face_bboxes)
        roi = []
        for idx, tbbox in enumerate(trans_bboxes):
            roi.append([0, *tbbox])
        if not roi:
            raise ValueError('No suitable candidate box, '
                             'try to modify the aspect ratio or cancel face detection')
        input_img = color_normalization_op(input_img)  # H W 3 RGB format
        id_out = self.eval_rois(input_img, roi)
        select_bboxes = []
        for id_ in id_out[:topK]:
            select_bboxes.append([int(round(item)) for item in source_bboxes[id_]])
        return select_bboxes
