from JPEG import JPEG
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import sys

if __name__=='__main__':
    
    image = Image.open('./test/kodim02.png')
    jpeg = JPEG(image, block_size=16, num_coeff=10, grayScale=False)
    encoded = jpeg.encoder()
    decoded_image = jpeg.decoder()
    mse, psnr = jpeg.get_MSE_and_PSNR()
    rmse = jpeg.getRMSE()
    compression_ratio = jpeg.getCompressionRatio()
    print(round(compression_ratio, 2), round(psnr, 2), round(rmse, 2))


    # # Test for Vary number of coefficient parameter sent
    # # CR = [511.91999999999973, 170.64000000000007, 85.44000000000003, 51.12000000000001, 34.08000000000002, 18.240000000000002]
    # # PSNR = [426.38, 478.38, 514.7299999999999, 547.03, 576.16, 612.56]
    # # CR = [639.0699999999999, 213.0, 106.54, 63.90999999999999, 42.629999999999995, 22.859999999999996]
    # # PSNR = [246.42, 278.44000000000005, 304.32000000000005, 320.19, 331.31, 343.18]
    # # params = [1, 3, 6, 10, 15, 28]

    # CR = [x / 10 for x in CR]
    # PSNR = [x / 10 for x in PSNR]
    # # print(CR)
    # # print(PSNR)
    # # sys.exit(0)

    # plt.plot(params, CR, label='Compression Ratio')
    # plt.plot(params, PSNR, label='PSNR')

    # plt.ylabel('y-axis')
    # plt.xlabel('Num-coeff')
    # plt.title('Average PSNR and Compression Ratio vs Num-coeff for 10 grayscale images')
    # plt.legend()
    # plt.show()
    #     # plt.savefig('results_grayscale_16/'+str(filename)+'.png')
    # # plt.close()

    # sys.exit(0)


    # directory = './test_images'
    # L = [1, 3, 6, 10, 15, 28]
    # CR = [0, 0, 0, 0, 0, 0]
    # PSNR = [0, 0, 0, 0, 0, 0]
    # for filename in os.listdir(directory):
        
    #     for i in range(6):
            
    #         # print(filename)
    #         f = os.path.join(directory, filename)
    #         # img = Image.open(f)
    #         img = Image.open(f)
    #         jpeg = JPEG(img, block_size=8, num_coeff=L[i], grayScale=True)
    #         encoded = jpeg.encoder()
    #         decoded_image = jpeg.decoder()
    #         mse, psnr = jpeg.get_MSE_and_PSNR()
    #         # rmse = jpeg.getRMSE()
    #         compression_ratio = jpeg.getCompressionRatio()
    #         # print(round(compression_ratio, 2), round(psnr, 2), round(rmse, 2))
    #         # RMSE.append(round(rmse, 2))
    #         # CR.append(round(compression_ratio, 2))
    #         # print(round(compression_ratio, 2), i)
    #         CR[i] += round(compression_ratio, 2)
    #         PSNR[i] += round(psnr, 2)
    #         # PSNR.append(round(psnr, 2))
    #         # print(round(psnr, 2), i)
    #         # # jpeg.display()
    #         print("running", i)

    #         # max_PSNR = max(PSNR)
    #         # normalized_PSNR = [round((x/max_PSNR)*100, 2) for x in PSNR]
    #         # # plt.plot(params, RMSE, label='RMSE')
    #         # L.append(i)
    #     # print(CR)
    #     # print(PSNR)
    #     # # plt.plot(L, PSNR)
    #     #     # # plt.plot(params, CR, label='Compression Ratio')
    #     #     # # plt.plot(params, PSNR, label='PSNR')
    #     #     # # plt.plot(params, normalized_PSNR, label='Normalized PSNR')
    #     # plt.ylabel('PSNR')
    #     # plt.xlabel('Block Size')
    #     # plt.title('Block Size vs PSNR')
    #     # plt.legend()
    #     # plt.show()
    #         # # plt.savefig('results_grayscale_16/'+str(filename)+'.png')
    #         # plt.close()
    # print(CR)
    # print(PSNR)



    # Test for Vary block size parameter sent
    # params = [1, 3, 6, 10, 15, 28]
    # img = Image.open('./test/kodim02.png')
    # for param in params:
    #     jpeg = JPEG(img, block_size=8, num_coeff=param, grayScale=False)
    #     encoded = jpeg.encoder()
    #     decoded_image = jpeg.decoder()
    #     jpeg.save('results/'+str(param)+'.png')

