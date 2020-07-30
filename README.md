# auto-crop

Image auto crop toolbox. Supports image cropping with any aspect ratio, based on [face-detection](https://pypi.org/project/face-detection/) and [GAIC: Grid-Anchor-based-Image-Cropping-Pytorch](https://github.com/lld533/Grid-Anchor-based-Image-Cropping-Pytorch). 

**This project only supports python3 and pytorch > 1.0**

## Contents

- [Setup](#Setup)
- [Demo](#Demo)
- [Different from GAIC](#Different-from-GAIC)
- [Reference](#Reference)
- [Citation](#Citation)

## Setup

You can choose to install via pypi or install it via source code.

### From Pypi

**auto-crop** can be installed directly through pypi

```shell
pip install auto-crop
```

### From Source Code

```shell
# colone repository
git clone https://github.com/lih627/autocrop.git
# install autocrop
cd /path/to/autocrop
python setup.py install
```

**Note** : If there is errors when compiling CPP and CUDA extensions, you can choose to compile CPP/CUDA api separately.

```shell
cd autocrop/model/rod_align
python setup.py install
cd ../roi_aligm
python setup.py install
```

## Demo

Here is a simple demo. From [`demo.py`](./demo.py)

First, build a cropper. Cropper contains [GAIC pretrained models](https://github.com/lld533/Grid-Anchor-based-Image-Cropping-Pytorch/tree/master/pretrained_model), and you can select load a [DSFD Face-Detecor](https://pypi.org/project/face-detection/) or not.

```python
from autocrop import cropper
autocropper = cropper.AutoCropper(model='mobilenetv2', # 'mobilenetv2' or 'shufflenetv2'
                                  cuda=True, # if GUDA is avaliable and True, Inference on GPU
                                  use_face_detector=True) # Use Face Detector to filter RoIs
```

Then, use cropper to crop RGB formate image. The selectable parameters are the number of cropping results, the aspect ration and whether to used face detection results to assist in generating RoIs. `crop_ret` is a list with size`topK x 4` ,  Each cropping result is encoded as `[xmin, ymin, xmax, ymax]` in pixel coordinate system.

```python
import cv2
# BGR to RGB
img = cv2.imread('imgs/demo.jpg')
img_ = img[:, :, (2, 1, 0)]
# get crop result
crop_ret = autocropper.crop(img_,
                            topK=1,
                            crop_height=1,
                            crop_width=1,
                            filter_face=True, # True: Crop result will not contain half face
                            single_face_center=True) # True: face in the crop result's width center
```

You can visualize the cropping results

```python
for bbox in crop_ret:
    r, g, b = int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (b, g, r))
cv2.imshow('ret', img)
cv2.waitKey()
```



## Different from GAIC

This project is mostly based on GAIC, and the modules are listed as follows:

<img src="https://github.com/lih627/autocrop/blob/master/misc/Pipeline.png?raw=true" alt="pipeline-w" style="zoom:50%;" />

It is slightly different from GAIC in practice, as shown below:

1. **We can specify any crop ratio**

GAIC supports RoIs with uncertain aspect ratios and several RoIs with fixed aspect ratios(`1:1, 4:3, 16:9`). In practical applications, image cropping needs to select the cropping area according to the fixed aspect ratio. I modified the code of the bboxes generation part. For RoIs evaluation, I used the GAIC pre-trained model.

2. **If there is only half a face in the bounding box, filter out the bounding box**

At the same time, in practical applications, when the distribution of people in the picture is not fixed, for example, when two people stand on the left and right sides of the picture, the RoI selected by GAIC may tear the human body. We adopt the face detection method to filter out some non-conformities. The required RoI will be evaluated after.

There is a comparison:

<img src="https://github.com/lih627/autocrop/blob/master/misc/face_filter.jpg?raw=true" alt="comparison with face detection - w150" style="zoom:50%;" />

3. **When the bounding box has only one face, the face should be in the middle of the box as much as possible**

We have added additional options when generating anchor boxes. If there is only one face in a RoI, use the RoI with the face in the middle of the RoI's width direction. see `autocrop/cropper.py` for details.

There is a comparison:

<img src="https://github.com/lih627/autocrop/blob/master/misc/face_filter2.jpg?raw=true" alt="comparison with face detection - w150" style="zoom:50%;" />

## Reference

1. [GAIC: Grid-Anchor-based-Image-Cropping-Pytorch](https://github.com/lld533/Grid-Anchor-based-Image-Cropping-Pytorch) MIT License
2. [DSFD-Pytorch-Inference](https://github.com/hukkelas/DSFD-Pytorch-Inference) [Apache-2.0 License](https://github.com/hukkelas/DSFD-Pytorch-Inference/blob/master/LICENSE)

## Citation

If you find this code useful, remember to cite the original authors:

For GAIC:

```
@inproceedings{zhang2019deep,
  title={Reliable and Efficient Image Cropping: A Grid Anchor based Approach},
  author={Zeng, Hui, Li, Lida， Cao, Zisheng and Zhang, Lei},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```

For DSFD:

```
@inproceedings{li2018dsfd,
  title={DSFD: Dual Shot Face Detector},
  author={Li, Jian and Wang, Yabiao and Wang, Changan and Tai, Ying and Qian, Jianjun and Yang, Jian and Wang, Chengjie and Li, Jilin and Huang, Feiyue},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```



