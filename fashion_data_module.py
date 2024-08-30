import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as transforms
from randaugment import RandAugment
from torch.utils.data import DataLoader
from q2l_labeller.data.cutmix import CutMixCollator


import sys
sys.path.append("C:/Users/ksrks/OneDrive - UOS/문서 - 인공지능 프로젝트")

from test import FashionDataset

class FashionDataModule(pl.LightningDataModule):
    """Datamodule for Lightning Trainer"""

    def __init__(
        self,
        data_dir,
        img_size,
        batch_size=128,
        num_workers=0,
        use_cutmix=False,
        cutmix_alpha=1.0,
    ) -> None:
        """_summary_

        Args:
            data_dir (str): Location of data.
            img_size (int): Desired size for transformed images.
            batch_size (int, optional): Dataloader batch size. Defaults to 128.
            num_workers (int, optional): Number of CPU threads to use. Defaults to 0.
            use_cutmix (bool, optional): Flag to enable Cutmix augmentation. Defaults to False.
            cutmix_alpha (float, optional): Defaults to 1.0.
        """
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_cutmix = use_cutmix
        self.cutmix_alpha = cutmix_alpha
        self.collator = torch.utils.data.dataloader.default_collate

    def prepare_data(self) -> None:
        """Loads metadata file and subsamples it if requested"""
        pass

    def setup(self, stage=None) -> None:
        """Creates train, validation, test datasets

        Args:
            stage (str, optional): Stage. Defaults to None.
        """
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        # mean=[0, 0, 0],
        # std=[1, 1, 1])

        train_transforms = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                RandAugment(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        test_transforms = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.train_set = FashionDataset(
            image_dir=(self.data_dir + "train_data"),
            # anno_path=(self.data_dir + "annotations/instances_val2017.json"),
            input_transform=train_transforms,
            csv_path=(self.data_dir + "final_train_image_labels_onehot.csv"),
        )
        self.val_set = FashionDataset(
            image_dir=(self.data_dir + "val_data"),
            # anno_path=(self.data_dir + "annotations/instances_val2017.json"),
            input_transform=test_transforms,
            csv_path=(self.data_dir + "final_validation_image_labels_onehot.csv"),
        )
    
        self.test_set = FashionDataset(
            image_dir=(self.data_dir + "test_data"),
            # anno_path=(self.data_dir + "annotations/instances_val2017.json"),
            input_transform=test_transforms,
            csv_path=(self.data_dir + "final_test_image_labels_onehot.csv"),
        )


        if self.use_cutmix:
            self.collator = CutMixCollator(self.cutmix_alpha)

    def get_num_classes(self):
        """Returns number of classes

        Returns:
            int: number of classes
        """
        return len(self.classes)

    def train_dataloader(self) -> DataLoader:
        """Creates and returns training dataloader

        Returns:
            DataLoader: Training dataloader
        """
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collator,
        )

    def val_dataloader(self) -> DataLoader:
        """Creates and returns validation dataloader

        Returns:
            DataLoader: Validation dataloader
        """
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Creates and returns test dataloader

        Returns:
            DataLoader: Test dataloader
        """
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
if __name__ == "__main__":
    img_data_dir = "C:/Users/ksrks/OneDrive - UOS/문서 - 인공지능 프로젝트/fashion_dataset_v2/"

    param_dict = {
    "backbone_desc":"resnet18",
    "conv_out_dim":512,
    "hidden_dim":256,
    "num_encoders":6,
    "num_decoders":6,
    "num_heads":8,
    "batch_size":32,
    "image_dim":224,
    "learning_rate":0.0001, 
    "momentum":0.9,
    "weight_decay":0.01, 
    "n_classes":5,
    "thresh":0.5,
    "use_cutmix":True,
    "use_pos_encoding":False,
    # "loss":"ASL"
}
    
    coco = FashionDataModule(
    img_data_dir,
    img_size=param_dict["image_dim"],
    batch_size=param_dict["batch_size"],
    num_workers=8,
    use_cutmix=param_dict["use_cutmix"],
    cutmix_alpha=1.0)
    coco.setup()
    param_dict["data"] = coco
    train_loader = coco.train_dataloader()
    val_loader = coco.val_dataloader()
    for batch in train_loader:
        # 여기서 batch는 DataLoader의 반환값이므로 각 미니배치의 데이터와 레이블을 가져올 수 있습니다.
        # print(batch)
        input, labels, img_name = batch

        # 원하는 작업 수행
        print(f"Batch size: {labels[0]}")

        break