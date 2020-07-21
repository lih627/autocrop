from autocrop import cropper
import cv2
import random

# create autocropper
autocropper = cropper.AutoCropper(model='mobilenetv2',
                                  cuda=True,
                                  use_face_detector=True)
# BGR to RGB
img = cv2.imread('imgs/demo.jpg')
img_ = img[:, :, (2, 1, 0)]

# get crop result
crop_ret = autocropper.crop(img_,
                            topK=1,
                            crop_height=2,
                            crop_width=5,
                            filter_face=True)
# save crop img
# bbox = crop_ret[0]
# cv2.imwrite('ret.jpg', img[bbox[1]: bbox[3] + 1, bbox[0]: bbox[2] + 1, :])
# show crop ret
for bbox in crop_ret:
    r, g, b = int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (b, g, r))

cv2.imshow('ret', img)
cv2.waitKey()
