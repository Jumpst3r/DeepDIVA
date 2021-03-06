import os

from unittest import TestCase
import PIL
from PIL import Image
from datasets.transform_library import transforms

from datasets.image_folder_segmentation_hisdb import find_classes, is_image_file, ImageFolder


class Test_get_item(TestCase):
    def setUp(self):
        self.test_dir = "/Users/tomasz/DeepDIVA/datasets/unit_tsts/test"
        self.pages_in_memory = 0
        self.crops_per_image = 0
        self.crop_size = 256
        self.test_ds = ImageFolder(self.test_dir, self.pages_in_memory, self.crops_per_image, self.crop_size)

    def test_length_of_epoch(self):
        # check init to see the formula how it was calculated
        # vertical crops: 20
        # horizontal crops: 26
        # number of test images: 10
        self.assertEqual(self.test_ds.__len__(), 26*26*10)

    def test_crops_per_image(self):
        self.assertEqual(self.test_ds.crops_per_image, 1989)

    def test_get_item(self):
        image_gt_transform = transforms.Compose([
            transforms.RandomTwinCrop(),
            transforms.ToTensorTwinImage()
        ])
        self.test_ds.transform = image_gt_transform

        index = 0 # value that is not used at all

        length_of_dataset = self.test_ds.__len__()

        for i in range(length_of_dataset):
            ((window_input_torch, (self.img_heigth, self.img_width), (x_position, y_position), image_name),
             one_hot_matrix) = self.test_ds.__getitem__(index, unittesting=True)

            print(str(image_name[:]) + " -> img_heigth: " + str(self.img_heigth) + " img_width: " + str(self.img_width)
                  + " /// x: " + str(x_position) + " y: " + str(y_position))

            # because of time constraints only manual testing performed
            # If you want to run this part that saves all the crops to disk for further analysis
            # make sure not to change the PIL.image.image into an torch array and not to do the one_hot transformation
            # uncomment: 3 lines in test_crop()

            # yolo_pass = os.path.join(self.test_dir, 'yolo')
            # if not os.path.isdir(yolo_pass):
            #     os.makedirs(yolo_pass)

            # window_input_torch.save(yolo_pass + "/img_" + str(i), "png")
            # one_hot_matrix.save(yolo_pass + "/gt_" + str(i), "png")
