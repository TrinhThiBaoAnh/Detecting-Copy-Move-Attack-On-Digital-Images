from pathlib import Path
from PIL import Image
import imageio
import tqdm

import numpy as np
import math
import time

from sklearn.decomposition import PCA

class Blocks(object):
    """
    Contains a single image block and handle the calculation of characteristic features
    """

    def __init__(self, grayscale_image_block, rgb_image_block, x_coordinate, y_coordinate, block_dimension):
        """
        Initializing the input image
        :param grayscale_image_block: grayscale image block
        :param rgb_image_block: rgb image block
        :param x_coordinate: x coordinate (upper-left)
        :param y_coordinate: y coordinate (upper-left)
        :return: None
        """
        self.image_grayscale = grayscale_image_block  # block of grayscale image
        self.image_grayscale_pixels = self.image_grayscale.load()

        if rgb_image_block is not None:
            self.image_rgb = rgb_image_block
            self.image_rgb_pixels = self.image_rgb.load()
            self.is_image_rgb = True
        else:
            self.is_image_rgb = False

        self.coordinate = (x_coordinate, y_coordinate)
        self.block_dimension = block_dimension

    def compute_block(self):
        """
        Create a representation of the image block
        :return: image block representation data
        """
        block_data_list = []
        block_data_list.append(self.coordinate)
        block_data_list.append(self.compute_characteristic_features(precision=4))
        block_data_list.append(self.compute_pca(precision=6))
        return block_data_list

    def compute_pca(self, precision):
        """
        Compute Principal Component Analysis from the image block
        :param precision: characteristic features precision
        :return: Principal Component from the image block
        """
        pca_module = PCA(n_components=1)
        if self.is_image_rgb:
            image_array = np.array(self.image_rgb)
            red_feature = image_array[:, :, 0]
            green_feature = image_array[:, :, 1]
            blue_feature = image_array[:, :, 2]

            concatenated_array = np.concatenate((red_feature, np.concatenate((green_feature, blue_feature), axis=0)), axis=0)
            pca_module.fit_transform(concatenated_array)
            principal_components = pca_module.components_
            precise_result = [round(element, precision) for element in list(principal_components.flatten())]
            return precise_result
        else:
            image_array = np.array(self.image_grayscale)
            pca_module.fit_transform(image_array)
            principal_components = pca_module.components_
            precise_result = [round(element, precision) for element in list(principal_components.flatten())]
            return precise_result

    def compute_characteristic_features(self, precision):
        """
        Compute 7 characteristic features from every image blocks
        :param precision: feature characteristic precision
        :return: None
        """

        characteristic_feature_list = []

        # variable to compute characteristic features
        c4_part1 = 0
        c4_part2 = 0
        c5_part1 = 0
        c5_part2 = 0
        c6_part1 = 0
        c6_part2 = 0
        c7_part1 = 0
        c7_part2 = 0

        """ Compute c1, c2, c3 according to the image block's colorspace """

        if self.is_image_rgb:
            sum_of_red_pixel_value = 0
            sum_of_green_pixel_value = 0
            sum_of_blue_pixel_value = 0
            for y_coordinate in range(0, self.block_dimension):  # compute sum of the pixel value
                for x_coordinate in range(0, self.block_dimension):
                    tmp_red, tmp_green, tmp_blue = self.image_rgb_pixels[x_coordinate, y_coordinate]
                    sum_of_red_pixel_value += tmp_red
                    sum_of_green_pixel_value += tmp_green
                    sum_of_blue_pixel_value += tmp_blue

            sum_of_pixels = self.block_dimension * self.block_dimension
            sum_of_red_pixel_value = sum_of_red_pixel_value / (sum_of_pixels)  # mean from each of the colorspaces
            sum_of_green_pixel_value = sum_of_green_pixel_value / (sum_of_pixels)
            sum_of_blue_pixel_value = sum_of_blue_pixel_value / (sum_of_pixels)

            characteristic_feature_list.append(sum_of_red_pixel_value)
            characteristic_feature_list.append(sum_of_green_pixel_value)
            characteristic_feature_list.append(sum_of_blue_pixel_value)

        else:
            characteristic_feature_list.append(0)
            characteristic_feature_list.append(0)
            characteristic_feature_list.append(0)

        """ Compute  c4, c5, c6 and c7 according to the pattern rule on the second paper"""
        for y_coordinate in range(0, self.block_dimension):  # compute the part 1 and part 2 of each feature characteristic
            for x_coordinate in range(0, self.block_dimension):
                # compute c4
                if y_coordinate <= self.block_dimension / 2:
                    c4_part1 += self.image_grayscale_pixels[x_coordinate, y_coordinate]
                else:
                    c4_part2 += self.image_grayscale_pixels[x_coordinate, y_coordinate]
                # compute c5
                if x_coordinate <= self.block_dimension / 2:
                    c5_part1 += self.image_grayscale_pixels[x_coordinate, y_coordinate]
                else:
                    c5_part2 += self.image_grayscale_pixels[x_coordinate, y_coordinate]
                # compute c6
                if x_coordinate - y_coordinate >= 0:
                    c6_part1 += self.image_grayscale_pixels[x_coordinate, y_coordinate]
                else:
                    c6_part2 += self.image_grayscale_pixels[x_coordinate, y_coordinate]
                # compute c7
                if x_coordinate + y_coordinate <= self.block_dimension:
                    c7_part1 += self.image_grayscale_pixels[x_coordinate, y_coordinate]
                else:
                    c7_part2 += self.image_grayscale_pixels[x_coordinate, y_coordinate]

        characteristic_feature_list.append(float(c4_part1) / float(c4_part1 + c4_part2))
        characteristic_feature_list.append(float(c5_part1) / float(c5_part1 + c5_part2))
        characteristic_feature_list.append(float(c6_part1) / float(c6_part1 + c6_part2))
        characteristic_feature_list.append(float(c7_part1) / float(c7_part1 + c7_part2))

        precise_result = [round(element, precision) for element in characteristic_feature_list]
        return precise_result
class Container(object):
    """
    Object to contains the computation result
    """

    def __init__(self):
        """
        List initialization
        :return: none
        """
        self.container = []
        return

    def get_length(self):
        """
        To return the current container's length
        :return: length of the container
        """
        return self.container.__len__()

    def append_block(self, newData):
        """
        Insert a data block to the container
        :param newData: data to be inserted into the block
        :return: None
        """
        self.container.append(newData)
        return

    def sort_by_features(self):
        """
        Sort all the container's data based on certain key
        :return: None
        """
        self.container = sorted(self.container, key=lambda x:(x[1], x[2]))
        return

    """
    Functions for debug purpose
    """
    def print_all_container(self):
        """
        Prints all the elements inside the container
        :return: None
        """
        for index in range(0, self.container.__len__()):
            print(self.container[index])
        return

    def print_container(self, count):
        """
        Prints certain elements inside the container
        :param count: amount to be printed
        :return: None
        """
        print(f"Element's index: {self.get_length()}")
        if count > self.get_length():
            self.print_all_container()
        else:
            for index in range(0, count):
                print(self.container[index])
        return
class ImageObject(object):
    """
    Object to contains a single image, then detects a fraud in it
    """

    def __init__(self, input_path, image_name, output_directory, block_dimension):
        """
        Constructor to initialize the algorithm's parameters
        :param input_path: image file directory
        :param image_name: image file name
        :param block_dimension: dimension block size (ex:32, 64, 128)
        :param output_directory: directory for detection results
        :return: None
        """
        print(image_name)
        print("Step 1 of 4: Object and variable initialization, ", end='')

        # image parameter
        self.image_output_directory = output_directory
        self.image_name = image_name
        self.image_data = Image.open(input_path)
        self.image_width, self.image_height = self.image_data.size  # height = vertical, width = horizontal

        if self.image_data.mode != 'L':  # L means grayscale
            self.is_rgb_image = True
            self.image_data = self.image_data.convert('RGB')
            rgb_image_pixels = self.image_data.load()
            self.image_grayscale = self.image_data.convert('L')  # creates a grayscale version of current image to be used later
            grayscale_image_pixels = self.image_grayscale.load()

            for y_coordinate in range(0, self.image_height):
                for x_coordinate in range(0, self.image_width):
                    red_pixel_value, green_pixel_value, blue_pixel_value = rgb_image_pixels[x_coordinate, y_coordinate]
                    grayscale_image_pixels[x_coordinate, y_coordinate] = int(0.299 * red_pixel_value) + int(
                        0.587 * green_pixel_value) + int(0.114 * blue_pixel_value)
        else:
            self.is_rgb_image = False
            self.image_data = self.image_data.convert('L')

        # algorithm's parameters from the first paper
        self.N = self.image_width * self.image_height
        self.block_dimension = block_dimension
        self.b = self.block_dimension * self.block_dimension
        self.Nb = (self.image_width - self.block_dimension + 1) * (self.image_height - self.block_dimension + 1)
        self.Nn = 2  # amount of neighboring block to be evaluated
        self.Nf = 188  # minimum treshold of the offset's frequency
        self.Nd = 50  # minimum treshold of the offset's magnitude

        # algorithm's parameters from the second paper
        self.P = (1.80, 1.80, 1.80, 0.0125, 0.0125, 0.0125, 0.0125)
        self.t1 = 2.80
        self.t2 = 0.02

        print(self.Nb, self.is_rgb_image)

        # container initialization to later contains several data
        self.features_container = Container()
        self.block_pair_container = Container()
        self.offset_dictionary = {}

    def run(self):
        """
        Run the created algorithm
        :return: None
        """

        # time logging (optional, for evaluation purpose)
        start_timestamp = time.time()
        self.compute()
        timestamp_after_computing = time.time()
        self.sort()
        timestamp_after_sorting = time.time()
        self.analyze()
        timestamp_after_analyze = time.time()
        image_result_path = self.reconstruct()
        timestamp_after_image_creation = time.time()

        print("Computing time :", timestamp_after_computing - start_timestamp, "second")
        print("Sorting time   :", timestamp_after_sorting - timestamp_after_computing, "second")
        print("Analyzing time :", timestamp_after_analyze - timestamp_after_sorting, "second")
        print("Image creation :", timestamp_after_image_creation - timestamp_after_analyze, "second")

        total_running_time_in_second = timestamp_after_image_creation - start_timestamp
        total_minute, total_second = divmod(total_running_time_in_second, 60)
        total_hour, total_minute = divmod(total_minute, 60)
        print("Total time    : %d:%02d:%02d second" % (total_hour, total_minute, total_second), '\n')
        return image_result_path

    def compute(self):
        """
        To compute the characteristic features of image block
        :return: None
        """
        print("Step 2 of 4: Computing characteristic features")

        image_width_overlap = self.image_width - self.block_dimension
        image_height_overlap = self.image_height - self.block_dimension

        if self.is_rgb_image:
            for i in tqdm.tqdm(range(0, image_width_overlap + 1, 1)):
                for j in range(0, image_height_overlap + 1, 1):
                    image_block_rgb = self.image_data.crop((i, j, i + self.block_dimension, j + self.block_dimension))
                    image_block_grayscale = self.image_grayscale.crop(
                        (i, j, i + self.block_dimension, j + self.block_dimension))
                    image_block = Blocks(image_block_grayscale, image_block_rgb, i, j, self.block_dimension)
                    self.features_container.append_block(image_block.compute_block())
        else:
            for i in range(image_width_overlap + 1):
                for j in range(image_height_overlap + 1):
                    image_block_grayscale = self.image_data.crop((i, j, i + self.block_dimension, j + self.block_dimension))
                    image_block = Blocks(image_block_grayscale, None, i, j, self.block_dimension)
                    self.features_container.append_block(image_block.compute_block())

    def sort(self):
        """
        To sort the container's elements
        :return: None
        """
        self.features_container.sort_by_features()

    def analyze(self):
        """
        To analyze pairs of image blocks
        :return: None
        """
        print("Step 3 of 4:Pairing image blocks")
        z = 0
        time.sleep(0.1)
        feature_container_length = self.features_container.get_length()

        for i in tqdm.tqdm(range(feature_container_length - 1)):
            j = i + 1
            result = self.is_valid(i, j)
            if result[0]:
                self.add_dictionary(self.features_container.container[i][0],
                                    self.features_container.container[j][0],
                                    result[1])
                z += 1

    def is_valid(self, first_block, second_block):
        """
        To check the validity of the image block pairs and each of the characteristic features,
        also compute its offset, magnitude, and absolut value.
        :param first_block: the first block
        :param second_block: the second block
        :return: is the pair of i and j valid?
        """

        if abs(first_block - second_block) < self.Nn:
            i_feature = self.features_container.container[first_block][1]
            j_feature = self.features_container.container[second_block][1]

            # check the validity of characteristic features according to the second paper
            if abs(i_feature[0] - j_feature[0]) < self.P[0]:
                if abs(i_feature[1] - j_feature[1]) < self.P[1]:
                    if abs(i_feature[2] - j_feature[2]) < self.P[2]:
                        if abs(i_feature[3] - j_feature[3]) < self.P[3]:
                            if abs(i_feature[4] - j_feature[4]) < self.P[4]:
                                if abs(i_feature[5] - j_feature[5]) < self.P[5]:
                                    if abs(i_feature[6] - j_feature[6]) < self.P[6]:
                                        if abs(i_feature[0] - j_feature[0]) + abs(i_feature[1] - j_feature[1]) + \
                                                abs(i_feature[2] - j_feature[2]) < self.t1:
                                            if abs(i_feature[3] - j_feature[3]) + abs(i_feature[4] - j_feature[4]) + \
                                                    abs(i_feature[5] - j_feature[5]) + \
                                                    abs(i_feature[6] - j_feature[6]) < self.t2:

                                                # compute the pair's offset
                                                i_coordinate = self.features_container.container[first_block][0]
                                                j_coordinate = self.features_container.container[second_block][0]

                                                # Non Absolute Robust Detection Method
                                                offset = (
                                                    i_coordinate[0] - j_coordinate[0],
                                                    i_coordinate[1] - j_coordinate[1]
                                                )

                                                # compute the pair's magnitude
                                                magnitude = np.sqrt(math.pow(offset[0], 2) + math.pow(offset[1], 2))
                                                if magnitude >= self.Nd:
                                                    return 1, offset
        return 0,

    def add_dictionary(self, first_coordinate, second_coordinate, pair_offset):
        """
        Add a pair of coordinate and its offset to the dictionary
        """
        if pair_offset in self.offset_dictionary:
            self.offset_dictionary[pair_offset].append(first_coordinate)
            self.offset_dictionary[pair_offset].append(second_coordinate)
        else:
            self.offset_dictionary[pair_offset] = [first_coordinate, second_coordinate]

    def reconstruct(self):
        """
        Reconstruct the image according to the fraud detectionr esult
        """
        print("Step 4 of 4: Image reconstruction")

        # create an array as the canvas of the final image
        groundtruth_image = np.zeros((self.image_height, self.image_width))
        lined_image = np.array(self.image_data.convert('RGB'))

        sorted_offset = sorted(self.offset_dictionary,
                          key=lambda key: len(self.offset_dictionary[key]),
                          reverse=True)

        is_pair_found = False

        for key in sorted_offset:
            if self.offset_dictionary[key].__len__() < self.Nf * 2:
                break

            if is_pair_found == False:
                print('Found pair(s) of possible fraud attack:')
                is_pair_found = True

            print(key, self.offset_dictionary[key].__len__())

            for i in range(self.offset_dictionary[key].__len__()):
                # The original image (grayscale)
                for j in range(self.offset_dictionary[key][i][1],
                               self.offset_dictionary[key][i][1] + self.block_dimension):
                    for k in range(self.offset_dictionary[key][i][0],
                                   self.offset_dictionary[key][i][0] + self.block_dimension):
                        groundtruth_image[j][k] = 255

        if is_pair_found == False:
            print('No pair of possible fraud attack found.')

        # creating a line edge from the original image (for the visual purpose)
        for x_coordinate in range(2, self.image_height - 2):
            for y_cordinate in range(2, self.image_width - 2):
                if groundtruth_image[x_coordinate, y_cordinate] == 255 and \
                        (groundtruth_image[x_coordinate + 1, y_cordinate] == 0 or
                         groundtruth_image[x_coordinate - 1, y_cordinate] == 0 or
                         groundtruth_image[x_coordinate, y_cordinate + 1] == 0 or
                         groundtruth_image[x_coordinate, y_cordinate - 1] == 0 or
                         groundtruth_image[x_coordinate - 1, y_cordinate + 1] == 0 or
                         groundtruth_image[x_coordinate + 1, y_cordinate + 1] == 0 or
                         groundtruth_image[x_coordinate - 1, y_cordinate - 1] == 0 or
                         groundtruth_image[x_coordinate + 1, y_cordinate - 1] == 0):

                    # creating the edge line, respectively left-upper, right-upper, left-down, right-down
                    if groundtruth_image[x_coordinate - 1, y_cordinate] == 0 and \
                            groundtruth_image[x_coordinate, y_cordinate - 1] == 0 and \
                            groundtruth_image[x_coordinate - 1, y_cordinate - 1] == 0:
                        lined_image[x_coordinate - 2:x_coordinate, y_cordinate, 1] = 255
                        lined_image[x_coordinate, y_cordinate - 2:y_cordinate, 1] = 255
                        lined_image[x_coordinate - 2:x_coordinate, y_cordinate - 2:y_cordinate, 1] = 255
                    elif groundtruth_image[x_coordinate + 1, y_cordinate] == 0 and \
                            groundtruth_image[x_coordinate, y_cordinate - 1] == 0 and \
                            groundtruth_image[x_coordinate + 1, y_cordinate - 1] == 0:
                        lined_image[x_coordinate + 1:x_coordinate + 3, y_cordinate, 1] = 255
                        lined_image[x_coordinate, y_cordinate - 2:y_cordinate, 1] = 255
                        lined_image[x_coordinate + 1:x_coordinate + 3, y_cordinate - 2:y_cordinate, 1] = 255
                    elif groundtruth_image[x_coordinate - 1, y_cordinate] == 0 and \
                            groundtruth_image[x_coordinate, y_cordinate + 1] == 0 and \
                            groundtruth_image[x_coordinate - 1, y_cordinate + 1] == 0:
                        lined_image[x_coordinate - 2:x_coordinate, y_cordinate, 1] = 255
                        lined_image[x_coordinate, y_cordinate + 1:y_cordinate + 3, 1] = 255
                        lined_image[x_coordinate - 2:x_coordinate, y_cordinate + 1:y_cordinate + 3, 1] = 255
                    elif groundtruth_image[x_coordinate + 1, y_cordinate] == 0 and \
                            groundtruth_image[x_coordinate, y_cordinate + 1] == 0 and \
                            groundtruth_image[x_coordinate + 1, y_cordinate + 1] == 0:
                        lined_image[x_coordinate + 1:x_coordinate + 3, y_cordinate, 1] = 255
                        lined_image[x_coordinate, y_cordinate + 1:y_cordinate + 3, 1] = 255
                        lined_image[x_coordinate + 1:x_coordinate + 3, y_cordinate + 1:y_cordinate + 3, 1] = 255

                    # creating the straigh line, respectively upper, down, left, right line
                    elif groundtruth_image[x_coordinate, y_cordinate + 1] == 0:
                        lined_image[x_coordinate, y_cordinate + 1:y_cordinate + 3, 1] = 255
                    elif groundtruth_image[x_coordinate, y_cordinate - 1] == 0:
                        lined_image[x_coordinate, y_cordinate - 2:y_cordinate, 1] = 255
                    elif groundtruth_image[x_coordinate - 1, y_cordinate] == 0:
                        lined_image[x_coordinate - 2:x_coordinate, y_cordinate, 1] = 255
                    elif groundtruth_image[x_coordinate + 1, y_cordinate] == 0:
                        lined_image[x_coordinate + 1:x_coordinate + 3, y_cordinate, 1] = 255

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        import os
        imageio.imwrite(os.path.join(self.image_output_directory, (timestamp + "_" + self.image_name)), groundtruth_image)
        imageio.imwrite(os.path.join(self.image_output_directory, (timestamp + "_lined_" + self.image_name)), lined_image)

        return os.path.join(self.image_output_directory, (timestamp + "_lined_" + self.image_name))
def detect(input_path, output_path, block_size=32):
    """
    Detects an image under a specific directory
    :param input_path: path to input image
    :param output_path: path to output folder
    :param block_size: the block size of the image pointer (eg. 32, 64, 128)
    The smaller the block size, the more accurate the result is, but takes more time, vice versa.
    :return: None
    """

    input_path = Path(input_path)
    filename = input_path.name
    output_path= Path(output_path)

    if not input_path.exists():
        print("Error: Source Directory did not exist.")
        exit(1)
    elif not output_path.exists():
        print("Error: Output Directory did not exist.")
        exit(1)

    single_image = ImageObject(input_path, filename, output_path, block_size)
    image_result_path = single_image.run()

    print("Done.")
    return image_result_path

from tkinter import *
from tkinter import messagebox as mbox
from PIL import Image, ImageTk
from tkinter import filedialog
class aFrame(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.imageName = ""
        self.initUI()

    def initUI(self):
        self.parent.title("Image Copy-Move Detection")
        # self.style.theme_use("default")
        self.pack(fill=BOTH, expand=1)

        quitButton = Button(self, text="Open File", command=self.onFilePicker)
        quitButton.place(x=10, y=10)

        printButton = Button(self, text="Detect", command=self.onDetect)
        printButton.place(x=10, y=40)

        self.textBoxFile = Text(self, state='disabled', width=80, height = 1)
        self.textBoxFile.place(x=90, y=10)

        self.textBoxLog = Text(self, state='disabled', width=40, height=3)
        self.textBoxLog.place(x=90, y=40)

        # absolute image widget
        imageLeft = Image.open("/home/baoanh/Desktop/Ky2Nam5/Xử lý ảnh/BTL/image-copy-move-detection/examples/Empty.png")
        imageLeftLabel = ImageTk.PhotoImage(imageLeft)
        self.labelLeft = Label(self, image=imageLeftLabel)
        self.labelLeft.image = imageLeftLabel
        self.labelLeft.place(x=5, y=100)

        imageRight = Image.open("/home/baoanh/Desktop/Ky2Nam5/Xử lý ảnh/BTL/image-copy-move-detection/examples/Empty.png")
        imageRightLabel = ImageTk.PhotoImage(imageRight)
        self.labelRight = Label(self, image=imageRightLabel)
        self.labelRight.image = imageRightLabel
        self.labelRight.place(x=525, y=100)

        self.centerWindow()

    def centerWindow(self):
            w = 1045
            h = 620

            sw = self.parent.winfo_screenwidth()
            sh = self.parent.winfo_screenheight()

            x = (sw - w)/2
            y = (sh - h)/2
            self.parent.geometry('%dx%d+%d+%d' % (w, h, x, y))

    def onFilePicker(self):

        ftypes = [('PNG Files', '*.png'), ('All files', '*')]
        choosedFile = filedialog.askopenfilename(initialdir='../testcase_image/', filetypes = ftypes)
        # dlg = Image.open(filepath)
        # choosedFile = dlg.show()

        if choosedFile != '':
            print(choosedFile)
            self.imageName = str(choosedFile).split("/")[-1]
            self.imagePath = str(choosedFile).replace(self.imageName, '')

            self.textBoxFile.config(state='normal')
            self.textBoxFile.delete('1.0', END)
            self.textBoxFile.insert(END, choosedFile)
            self.textBoxFile.config(state='disabled')

            newImageLeft = Image.open(choosedFile)
            imageLeftLabel = ImageTk.PhotoImage(newImageLeft)
            self.labelLeft = Label(self, image=imageLeftLabel)
            self.labelLeft.image = imageLeftLabel
            self.labelLeft.place(x=5, y=100)

            imageRight = Image.open("/home/baoanh/Desktop/Ky2Nam5/Xử lý ảnh/BTL/image-copy-move-detection/examples/Empty.png")
            imageRightLabel = ImageTk.PhotoImage(imageRight)
            self.labelRight = Label(self, image=imageRightLabel)
            self.labelRight.image = imageRightLabel
            self.labelRight.place(x=525, y=100)

        pass

    def onDetect(self):
        if self.imageName == "":
            mbox.showerror("Error", 'No image selected\nSelect an image by clicking "Open File"')
        else:

            self.textBoxLog.config(state='normal')
            self.textBoxLog.insert(END, "Detecting: "+self.imageName+"\n")
            self.textBoxLog.see(END)
            self.textBoxLog.config(state='disabled')

            imageResultPath = detect(self.imagePath + self.imageName, '/home/baoanh/Desktop/Ky2Nam5/Xử lý ảnh/BTL/image-copy-move-detection/output', block_size=32)
            newImageRight = Image.open(imageResultPath)
            imageRightLabel = ImageTk.PhotoImage(newImageRight)
            self.labelRight = Label(self, image=imageRightLabel)
            self.labelRight.image = imageRightLabel
            self.labelRight.place(x=525, y=100)

            self.textBoxLog.config(state='normal')
            self.textBoxLog.insert(END, "Done.")
            self.textBoxLog.see(END)
            self.textBoxLog.config(state='disabled')

if __name__ == '__main__':
    root = Tk()
    app = aFrame(root)
    root.mainloop()

