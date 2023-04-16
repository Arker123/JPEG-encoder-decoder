from JPEG import JPEG
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import numpy as np

if __name__=='__main__':
    jpeg = JPEG(block_size=8, num_coeff=64)
    img = Image.open('./test/kodim01.png')
    # print(type(np.asarray(img)))
    # cv2.imshow('ff',img)
    # img.show()
    # img.show()
    encoded = jpeg.encoder(img)
    # for enc in encoded:
    #     print(enc)
    decoded_image = jpeg.decoder(encoded)

    # plt.imshow(decoded_image)
    # plt.show()
    img = np.asarray(img)
    # print(img)
    # decoded_image = np.asarray(decoded_image)
    rmse = jpeg.getRMSE(img, decoded_image, decoded_image.shape[0], decoded_image.shape[1])
    print(rmse)