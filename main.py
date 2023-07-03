from PIL import Image
from matplotlib import pyplot as plt

from JPEG import JPEG

if __name__ == "__main__":
    image = Image.open("./test/kodim02.png")
    jpeg = JPEG(image, block_size=16, num_coeff=10, grayScale=False)
    encoded = jpeg.encoder()
    decoded_image = jpeg.decoder()
    mse, psnr = jpeg.get_MSE_and_PSNR()
    rmse = jpeg.getRMSE()
    compression_ratio = jpeg.getCompressionRatio()
    print(round(compression_ratio, 2), round(psnr, 2), round(rmse, 2))
