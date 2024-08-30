import os

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
from tqdm import tqdm
import csv
from randaugment import RandAugment
os.environ['KMP_DUPLICATE_LIB_OK']='True'

category_map = {
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
}


class FashionDataset(data.Dataset):
    """Custom dataset that will load the COCO 2014 dataset and annotations

    This module will load the basic files as provided here: https://cocodataset.org/#download
    If the labels file does not exist yet, it will be created with the included
    helper functions. This class was largely taken from Shilong Liu's repo at
    https://github.com/SlongLiu/query2labels/blob/main/lib/dataset/cocodataset.py.

    Attributes:
        coco (torchvision dataset): Dataset containing COCO data.
        category_map (dict): Mapping of category names to indices.
        input_transform (list of transform objects): List of transforms to apply.
        labels_path (str): Location of labels (if they exist).
        used_category (int): Legacy var.
        labels (list): List of labels.

    """

    def __init__(
        self,
        image_dir,
        # anno_path,
        input_transform=None,
        csv_path=None,
        used_category=-1,
    ):
        """Initializes dataset

        Args:
            image_dir (str): Location of COCO images.
            anno_path (str): Location of COCO annotation files.
            input_transform (list of Transform objects, optional): List of transforms to apply.  Defaults to None.
            labels_path (str, optional): Location of labels. Defaults to None.
            used_category (int, optional): Legacy var. Defaults to -1.
        """
        
        self.category_map = category_map
        self.input_transform = input_transform
        # self.labels_path = labels_path
        self.used_category = used_category
        self.image_root = image_dir


        # CSV 파일 경로
        self.csv_file_path = csv_path

        # CSV 파일 읽기
        with open(self.csv_file_path, 'r') as file:
            # CSV 파일을 읽어들이는 reader 객체 생성
            reader = csv.reader(file)

            # 이러면 csv파일을 다 읽어와서 file descriptor가 맨 끝에 가있음
            self.data = list(reader)
            self.data_size = len(self.data)

            # 각 행의 값을 저장할 리스트
            data_list = []

            # 각 행에 대해 반복
            for row in self.data:
                
                # 행의 두 번째 열부터 마지막 열까지의 값을 추출하여 리스트에 추가
                values = list(map(float, row[1:]))
                data_list.append(values)

        # 리스트를 NumPy 배열로 변환
        self.labels = np.array(data_list).astype(np.float64)
    def getLabelVector(self, categories):
        """Creates multi-hot vector for item labels

        Args:
            categories (list): List of categories matching an item

        Returns:
            ndarray: Multi-hot vector for item labels
        """
        label = np.zeros(5)
        for c in categories:
            index = self.category_map[str(c)] - 1
            label[index] = 1.0  # / label_num
        return label


    def __getitem__(self, index):
        img_name = os.path.join(self.image_root, self.data[index][0])
        image = Image.open(img_name).convert('RGB')
        if self.input_transform:
            input = self.input_transform(image)
        return input, self.labels[index], img_name

  

    def __len__(self):
        return self.data_size

  





if __name__ == "__main__":
    current_directory = os.getcwd()
    print("현재 작업 디렉토리:", current_directory)
