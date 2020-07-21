import torch
import torch.nn as nn
import torch.nn.init as init
from torch.hub import load_state_dict_from_url

from .roi_align.modules.roi_align import RoIAlignAvg, RoIAlign
from .rod_align.modules.rod_align import RoDAlignAvg, RoDAlign
from .moblienetv2 import mobilenetv2
from .shufflenetv2 import shufflenetv2

MODEL_URL = {'shufflenetv2':
                 'https://github.com/lih627/Grid-Anchor-based-Image-Cropping-Pytorch/raw/master/pretrained_model/shufflenet_0.615_0.568_0.548_0.520_0.785_0.755_0.738_0.713_0.774_0.801.pth',
             'mobilenetv2':
                 'https://github.com/lih627/Grid-Anchor-based-Image-Cropping-Pytorch/raw/master/pretrained_model/mobilenet_0.625_0.583_0.553_0.525_0.785_0.762_0.748_0.723_0.783_0.806.pth'}


def fc_layers(reddim=32, alignsize=8):
    conv1 = nn.Sequential(nn.Conv2d(reddim, 768, kernel_size=alignsize, padding=0), nn.BatchNorm2d(768),
                          nn.ReLU(inplace=True))
    conv2 = nn.Sequential(nn.Conv2d(768, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
    dropout = nn.Dropout(p=0.5)
    conv3 = nn.Conv2d(128, 1, kernel_size=1)
    layers = nn.Sequential(conv1, conv2, dropout, conv3)
    return layers


class mobilenetv2_base(nn.Module):

    def __init__(self):
        super(mobilenetv2_base, self).__init__()

        model = mobilenetv2(width_mult=1.0)

        self.feature3 = nn.Sequential(model.features[:7])
        self.feature4 = nn.Sequential(model.features[7:14])
        self.feature5 = nn.Sequential(model.features[14:])

    def forward(self, x):
        # return self.feature(x)
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5


class shufflenetv2_base(nn.Module):

    def __init__(self):
        super(shufflenetv2_base, self).__init__()

        model = shufflenetv2(width_mult=1.0)
        self.feature3 = nn.Sequential(model.conv1, model.maxpool, model.features[:4])
        self.feature4 = nn.Sequential(model.features[4:12])
        self.feature5 = nn.Sequential(model.features[12:])

    def forward(self, x):
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5


class crop_model_multi_scale_shared(nn.Module):
    def __init__(self, alignsize=8, reddim=32, model=None, downsample=4):
        super(crop_model_multi_scale_shared, self).__init__()

        if model == 'shufflenetv2':
            self.Feat_ext = shufflenetv2_base()
            self.DimRed = nn.Conv2d(812, reddim, kernel_size=1, padding=0)
        elif model == 'mobilenetv2':
            self.Feat_ext = mobilenetv2_base()
            self.DimRed = nn.Conv2d(448, reddim, kernel_size=1, padding=0)
        else:
            raise ValueError("{} not supptored".format(model))

        self.downsample2 = nn.UpsamplingBilinear2d(scale_factor=1.0 / 2.0)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.RoIAlign = RoIAlignAvg(alignsize, alignsize, 1.0 / 2 ** downsample)
        self.RoDAlign = RoDAlignAvg(alignsize, alignsize, 1.0 / 2 ** downsample)
        self.FC_layers = fc_layers(reddim * 2, alignsize)

    def forward(self, im_data, boxes):

        f3, f4, f5 = self.Feat_ext(im_data)
        cat_feat = torch.cat((self.downsample2(f3), f4, 0.5 * self.upsample2(f5)), 1)
        red_feat = self.DimRed(cat_feat)
        RoI_feat = self.RoIAlign(red_feat, boxes)
        RoD_feat = self.RoDAlign(red_feat, boxes)
        final_feat = torch.cat((RoI_feat, RoD_feat), 1)
        prediction = self.FC_layers(final_feat)
        return prediction

    def _init_weights(self):
        print('Initializing weights...')
        self.DimRed.apply(weights_init)
        self.FC_layers.apply(weights_init)


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def build_crop_model(model='mobelnetv2',
                     cuda=False,
                     model_path=None,
                     alignsize=9, reddim=8, downsample=4):
    crop_model = crop_model_multi_scale_shared(alignsize=alignsize,
                                               reddim=reddim,
                                               model=model,
                                               downsample=downsample)
    mp_location = torch.device('cuda') if cuda else torch.device('cpu')
    if len(model_path) == 0:
        try:
            state_dict = load_state_dict_from_url(MODEL_URL[model],
                                                  map_location=mp_location,
                                                  progress=True)
        except Exception as e:
            raise e('Try to download pretrained model from \n https://github.com/lih627/Grid-Anchor-based-Image-Cropping-Pytorch/tree/master/pretrained_model'
                    ' \nand set model_path')

    else:
        state_dict = torch.load(model_path, map_location=mp_location)
    crop_model.load_state_dict(state_dict)

    return crop_model
