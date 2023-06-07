from paddle.io import Dataset
import os
from paddle.vision import image_load


class ImageNet2012Dataset(Dataset):
    """Build ImageNet2012 dataset
    This class gets train/val imagenet datasets, which loads transfomed data and labels.
    Attributes:
        file_folder: path where imagenet images are stored
        transform: preprocessing ops to apply on image
        img_path_list: list of full path of images in whole dataset
        label_list: list of labels of whole dataset
    """

    def __init__(self, file_folder, mode="train", transform=None):
        """Init ImageNet2012 Dataset with dataset file path, mode(train/val), and transform"""
        super(ImageNet2012Dataset, self).__init__()
        assert mode in ["train", "val"]
        self.file_folder = file_folder
        self.transform = transform
        self.img_path_list = []
        self.label_list = []

        if mode == "train":
            self.list_file = os.path.join(self.file_folder, "train_list.txt")
        else:
            self.list_file = os.path.join(self.file_folder, "val_list.txt")

        with open(self.list_file, 'r') as infile:
            for line in infile:
                img_path = line.strip().split()[0]
                img_label = int(line.strip().split()[1])
                self.img_path_list.append(os.path.join(self.file_folder, img_path))
                self.label_list.append(img_label)
        print(f'----- Imagenet2012 image {mode} list len = {len(self.label_list)}')

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        data = image_load(self.img_path_list[index]).convert('RGB')
        if isinstance(self.transform, (list, tuple)):
            data = [trans(data) for trans in self.transform]
        else:
            data = self.transform(data)
        label = self.label_list[index]

        return data, label
