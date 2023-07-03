import sys
import math

import cv2
import numpy as np
from matplotlib import pyplot as plt


class JPEG:
    """Class to encode and decode an image using JPEG"""

    def __init__(self, image, block_size=8, num_coeff=None, grayScale=False, Qmatrix=None):
        """Initialize the class"""
        self.image = image
        self.block_size = block_size

        # Quantization matrix
        q_matrix = np.array(
            [
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99],
            ]
        )
        scaling_factor = block_size / 8
        scaled_q_matrix = cv2.resize(
            q_matrix.astype("float32"), None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC
        )

        if Qmatrix is None:
            self.q_matrix = scaled_q_matrix.astype("int32")
        else:
            self.q_matrix = Qmatrix

        self.num_coeff = num_coeff
        self.copy_img = image

        self.grayScale = grayScale

        if num_coeff is not None:
            if block_size**2 < num_coeff:
                print("Number of coefficients cannot be greater than block size")
                sys.exit(1)

    def encoder(self):
        """
        Encode the image using JPEG
        :return: The encoded image
        """
        image = self.image
        nparray = np.asarray(image)

        new_width = math.ceil(self.image.size[0] / self.block_size) * self.block_size
        new_height = math.ceil(self.image.size[1] / self.block_size) * self.block_size

        new_height = new_height - self.image.size[1]
        new_width = new_width - self.image.size[0]

        if self.grayScale:
            img = np.pad(nparray, ((0, new_height), (0, new_width)), mode="constant")
        else:
            r, g, b = cv2.split(nparray)

            r = np.pad(r, ((0, new_height), (0, new_width)), mode="constant")
            g = np.pad(g, ((0, new_height), (0, new_width)), mode="constant")
            b = np.pad(b, ((0, new_height), (0, new_width)), mode="constant")

            img = cv2.merge((r, g, b))

        nparray = img

        self.shape = nparray.shape

        if self.grayScale:
            image = np.float32(np.int32(nparray))
            blocks = self.split_into_blocks_grayscale(image, self.block_size)
            encoded_input = []

            for block in blocks:
                dct_image = cv2.dct(block, cv2.DCT_INVERSE)
                out = np.divide(dct_image, self.q_matrix)
                out = np.int32(out)
                encoded_input.append(out)

            output_matrix = [self.zig_zag_array(block, self.num_coeff) for block in encoded_input]

            self.encoded_image = output_matrix
            return self.encoded_image
        else:
            yuv_image = self.RGB2YUV(nparray)

            image = np.float32(np.int32(yuv_image))
            blocks = self.split_into_blocks_colored(image, self.block_size)
            encoded_input = []

            for block in blocks:
                dct_image = cv2.dct(block, cv2.DCT_INVERSE)
                out = np.divide(dct_image, self.q_matrix)
                out = np.int32(out)
                encoded_input.append(out)

            output_matrix = [self.zig_zag_array(block, self.num_coeff) for block in encoded_input]

            self.encoded_image = output_matrix
            return self.encoded_image

    def decoder(self):
        """
        Decode the image using JPEG
        :return: The decoded image
        """
        encoded_input = self.encoded_image
        decoded_input = []

        for matrix in encoded_input:
            reformed_matrix = self.reconstruct_zig_zag(matrix, self.num_coeff, self.block_size)
            decoded_input.append(reformed_matrix)

        decoded_output = []

        for inp in decoded_input:
            tmp_image = np.multiply(inp, self.q_matrix)
            tmp_image = np.float32(tmp_image)
            img1 = cv2.idct(tmp_image)
            decoded_output.append(img1)

        if self.grayScale:
            new_img = np.zeros((self.shape[0], self.shape[1]))

            ct = 0

            for i in range(0, self.shape[0], self.block_size):
                for j in range(0, self.shape[1], self.block_size):
                    new_img[i : i + self.block_size, j : j + self.block_size] = decoded_output[ct]
                    ct += 1

            self.decoded_image = np.int32(new_img)
            return self.decoded_image
        else:
            new_img = np.zeros((self.shape[0], self.shape[1], self.shape[2]))

            ct = 0

            for i in range(0, self.shape[0], self.block_size):
                for j in range(0, self.shape[1], self.block_size):
                    for k in range(0, self.shape[2]):
                        new_img[i : i + self.block_size, j : j + self.block_size, k] = decoded_output[ct]
                        ct += 1

            self.decoded_image = np.int32(self.YUV2RGB(new_img))
            return self.decoded_image

    def split_into_blocks_colored(self, image, block_size):
        """
        Split the image into blocks of size block_size for color images
        :param image: The image to be split
        :param block_size: The size of the block
        :return: A list of blocks
        """

        tiles = []

        for i in range(0, image.shape[0], block_size):
            for j in range(0, image.shape[1], block_size):
                for k in range(0, image.shape[2]):
                    tile = image[i : i + block_size, j : j + block_size, k]
                    tiles.append(tile)

        return tiles

    def split_into_blocks_grayscale(self, image, block_size):
        """
        Split the image into blocks of size block_size for grayscale images
        :param image: The image to be split
        :param block_size: The size of the block
        :return: A list of blocks
        """

        tiles = []

        for i in range(0, image.shape[0], block_size):
            for j in range(0, image.shape[1], block_size):
                tile = image[i : i + block_size, j : j + block_size]
                tiles.append(tile)

        return tiles

    def RGB2YUV(self, rgb_image):
        """
        Convert the image from RGB to YUV
        """
        matrix = np.array(
            [[0.29900, -0.147108, 0.614777], [0.58700, -0.288804, -0.514799], [0.11400, 0.435912, -0.099978]]
        )
        yuv_image = np.dot(rgb_image, matrix)
        yuv_image[:, :, 1:] += 0.5
        return yuv_image

    def YUV2RGB(self, yuv_image):
        """
        Convert the image from YUV to RGB
        """
        matrix = np.array(
            [
                [1.000, 1.000, 1.000],
                [0.000, -0.394, 2.032],
                [1.140, -0.581, 0.000],
            ]
        )
        yuv_image[:, :, 1:] -= 0.5
        rgb_image = np.dot(yuv_image, matrix)
        return rgb_image

    def zig_zag_array(self, array, n=None):
        """
        Zig-zag array
        :param array: The array to be zig-zagged
        :param n: The number of elements to be zig-zagged
        :return: The zig-zagged array
        """

        shape = np.array(array).shape

        flag = False
        if n == None:
            flag = True
            n = self.block_size * self.block_size
        res = []

        (j, i) = (0, 0)
        direction = "r"  # {'r': right, 'd': down, 'ur': up-right, 'dl': down-left}
        for subel_num in range(1, n + 1):
            res.append(array[j][i])
            if direction == "r":
                i += 1
                if j == shape[0] - 1:
                    direction = "ur"
                else:
                    direction = "dl"
            elif direction == "dl":
                i -= 1
                j += 1
                if j == shape[0] - 1:
                    direction = "r"
                elif i == 0:
                    direction = "d"
            elif direction == "d":
                j += 1
                if i == 0:
                    direction = "ur"
                else:
                    direction = "dl"
            elif direction == "ur":
                i += 1
                j -= 1
                if i == shape[1] - 1:
                    direction = "d"
                elif j == 0:
                    direction = "r"

        if flag:
            res[:] = np.trim_zeros(res)

        return res

    def reconstruct_zig_zag(self, array, n, block_size):
        """
        Reconstruct the zig-zagged array
        :param array: The zig-zagged array
        :param n: The number of elements to be reconstructed
        :param block_size: The size of the block
        :return: The reconstructed array
        """

        result = np.int32(np.zeros([block_size, block_size]))

        shape = (block_size, block_size)

        if n == None:
            n = len(array)

        (j, i) = (0, 0)
        direction = "r"
        for subel_num in range(1, n + 1):
            result[j][i] = array[subel_num - 1]
            if direction == "r":
                i += 1
                if j == shape[0] - 1:
                    direction = "ur"
                else:
                    direction = "dl"
            elif direction == "dl":
                i -= 1
                j += 1
                if j == shape[0] - 1:
                    direction = "r"
                elif i == 0:
                    direction = "d"
            elif direction == "d":
                j += 1
                if i == 0:
                    direction = "ur"
                else:
                    direction = "dl"
            elif direction == "ur":
                i += 1
                j -= 1
                if i == shape[1] - 1:
                    direction = "d"
                elif j == 0:
                    direction = "r"

        return result

    def getRMSE(self):
        """
        Calculate the RMSE
        """

        inp = np.asarray(self.copy_img)
        out = self.decoded_image
        width = self.decoded_image.shape[0]
        height = self.decoded_image.shape[1]
        rmse = 0

        if self.grayScale:
            for i in range(0, inp.shape[0]):
                for j in range(0, inp.shape[1]):
                    val = abs(inp[i][j] - out[i][j])
                    val = val**2
                    rmse += val
        else:
            for k in range(0, len(inp)):
                for i in range(0, inp[k].shape[0]):
                    for j in range(0, inp[k].shape[1]):
                        val = abs(inp[k][i][j] - out[k][i][j])
                        val = val**2
                        rmse += val

        rmse = rmse / (width * height)
        rmse = math.sqrt(rmse)
        return rmse

    def get_MSE_and_PSNR(self):
        """
        Calculate the MSE and PSNR
        """

        inp = np.asarray(self.copy_img)
        out = self.decoded_image
        width = self.decoded_image.shape[0]
        height = self.decoded_image.shape[1]
        mse = 0

        if self.grayScale:
            for i in range(0, inp.shape[0]):
                for j in range(0, inp.shape[1]):
                    val = abs(inp[i][j] - out[i][j])
                    val = val**2
                    mse += val
        else:
            for k in range(0, len(inp)):
                for i in range(0, inp[k].shape[0]):
                    for j in range(0, inp[k].shape[1]):
                        val = abs(inp[k][i][j] - out[k][i][j])
                        val = val**2
                        mse += val

        mse = mse / (width * height)
        if mse == 0:
            psnr = float("inf")
        else:
            psnr = 20 * math.log10(255 / math.sqrt(mse))
        return mse, psnr

    def write_encoded_data(self, file):
        """Write encoded data to text file"""
        with open(file, "w") as f:
            for i in range(len(self.encoded_image)):
                f.write(str(self.encoded_image[i]) + " ")

    def display(self):
        """Display image"""

        if self.grayScale:
            plt.imshow(np.asarray(self.decoded_image), cmap="gray")
        else:
            plt.imshow(np.asarray(self.decoded_image))
        plt.show()

    def save(self, file):
        """Save image"""
        if self.grayScale:
            plt.imshow(np.asarray(self.decoded_image), cmap="gray")
        else:
            plt.imshow(np.asarray(self.decoded_image))
        plt.savefig(file)

    def getCompressionRatio(self):
        """Compression Ratio"""
        initial = self.copy_img.size[0] * self.copy_img.size[1] * 8
        l = 0
        for i in self.encoded_image:
            for j in i:
                l += 1
        final = l * 8
        return initial / final
