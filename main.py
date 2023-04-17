from JPEG import JPEG
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import numpy as np

if __name__=='__main__':
    
    img = Image.open('./test_images/lena.tif')
    jpeg = JPEG(img, block_size=8, num_coeff=64, grayScale=True)
    # print(type(np.asarray(img)))
    # cv2.imshow('ff',img)
    # img.show()
    # img.show()
    encoded = jpeg.encoder()
    # for enc in encoded:
    #     print(enc)
    decoded_image = jpeg.decoder()

    # plt.imshow(decoded_image)
    # plt.show()
    # img = np.asarray(img)
    # print(img)
    # decoded_image = np.asarray(decoded_image)

    rmse = jpeg.getRMSE()

    # jpeg.write_encoded_data('./encoded.txt')
    print(rmse)
    # jpeg.display()