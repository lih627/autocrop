from autocrop import cropper
import cv2
import random

# create autocropper
autocropper = cropper.AutoCropper(model='mobilenetv2',
                                  cuda=True,
                                  use_face_detector=True)
# BGR to RGB
img = cv2.imread('imgs/test2.jpg')
img_ = img[:, :, (2, 1, 0)]

# get crop result
crop_ret = autocropper.crop(img_,
                            topK=3,
                            crop_height=5,
                            crop_width=3,
                            filter_face=True,
                            single_face_center=True)
# save crop img
for idx, bbox in enumerate(crop_ret):
    cv2.imwrite('ret_{}.jpg'.format(idx), img[bbox[1]: bbox[3] + 1, bbox[0]: bbox[2] + 1, :])
# show crop ret
for bbox in crop_ret:
    r, g, b = int(random.random() * 255), int(random.random() * 255), int(random.random() * 255)
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (b, g, r))

cv2.imshow('ret', img)
cv2.waitKey()
cv2.destroyWindow('ret')