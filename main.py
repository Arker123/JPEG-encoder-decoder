from JPEG import JPEG
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os

if __name__=='__main__':
    
    # img = Image.open('./test_images/lena.tif')
    # jpeg = JPEG(img, block_size=8, num_coeff=32, grayScale=True)
    # # print(type(np.asarray(img)))
    # # cv2.imshow('ff',img)
    # # img.show()
    # # img.show()
    # encoded = jpeg.encoder()
    # # for enc in encoded:
    # #     print(enc)
    # decoded_image = jpeg.decoder()

    # # plt.imshow(decoded_image)
    # # plt.show()
    # # img = np.asarray(img)
    # # print(img)
    # # decoded_image = np.asarray(decoded_image)

    # # rmse = jpeg.getRMSE()

    # # mse, psnr = jpeg.get_MSE_and_PSNR()
    # # print(mse, psnr)

    # # # jpeg.write_encoded_data('./encoded.txt')
    # # print(rmse)

    # print(jpeg.getCompressionRatio())

    # # for im
    # # jpeg.display()
    
    # # Color image PSNR Table
    # directory = './test'
    # for filename in os.listdir(directory):
    #     f = os.path.join(directory, filename)
    #     # Checking if it is a file
    #     if os.path.isfile(f):
    #         img = Image.open(f)
    #         jpeg = JPEG(img, block_size=8, num_coeff=32, grayScale=False)
    #         encoded = jpeg.encoder()
    #         decoded_image = jpeg.decoder()
    #         mse, psnr = jpeg.get_MSE_and_PSNR()
    #         print(filename, psnr)

    # # Vary number of coefficient parameter sent
    

    # directory = './test_images'
    # for filename in os.listdir(directory):
    #     params = [1, 3, 6, 10, 15, 28]
    #     RMSE = []
    #     CR = []
    #     PSNR = []
    #     print(filename)
    #     f = os.path.join(directory, filename)
    #     for param in params:
    #         img = Image.open(f)
    #         # img = Image.open('./test/kodim02.png')
    #         jpeg = JPEG(img, block_size=16, num_coeff=param, grayScale=True)
    #         encoded = jpeg.encoder()
    #         decoded_image = jpeg.decoder()
    #         mse, psnr = jpeg.get_MSE_and_PSNR()
    #         rmse = jpeg.getRMSE()
    #         compression_ratio = jpeg.getCompressionRatio()
    #         print(param, round(compression_ratio, 2), round(psnr, 2), round(rmse, 2))
    #         RMSE.append(round(rmse, 2))
    #         CR.append(round(compression_ratio, 2))
    #         PSNR.append(round(psnr, 2))

    #     max_PSNR = max(PSNR)
    #     normalized_PSNR = [round((x/max_PSNR)*100, 2) for x in PSNR]
    #     plt.plot(params, RMSE, label='RMSE')
    #     plt.plot(params, CR, label='Compression Ratio')
    #     plt.plot(params, PSNR, label='PSNR')
    #     plt.plot(params, normalized_PSNR, label='Normalized PSNR')
    #     plt.xlabel('Number of Coefficients')
    #     plt.ylabel('y - axis')
    #     plt.title('RMSE, Compression Ratio and PSNR vs Number of Coefficients')
    #     plt.legend()
    #     plt.savefig('results_grayscale_16/'+str(filename)+'.png')
    #     plt.close()

    # Vary block size parameter sent
    params = [1, 3, 6, 10, 15, 28]
    img = Image.open('./test/kodim02.png')
    for param in params:
        jpeg = JPEG(img, block_size=8, num_coeff=param, grayScale=False)
        encoded = jpeg.encoder()
        decoded_image = jpeg.decoder()
        jpeg.save('results/'+str(param)+'.png')
        # jpeg.display()
        # im = Image.fromarray(decoded_image, "RGB")
        # im.save('results/'+str(param)+'.png')
        # cv2.imwrite('results/'+str(param)+'.png', decoded_image)
        # print(decoded_image)
        # img = Image.fromarray(np.asarray(decoded_image), "RGB")
        # img.save('results'+str(param)+'.png')
