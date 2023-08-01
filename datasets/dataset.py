from xml.etree.ElementTree import TreeBuilder
from torchvision import transforms
from torch.utils.data import Dataset
from .data_utils import get_onehot
from .augmentation.randaugment import RandAugment

import torchvision
from PIL import Image
import numpy as np
import copy


class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 alg,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 is_ulb=False,
                 strong_transform=None,
                 onehot=False,
                 *args, **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        self.alg = alg
        self.data = data
        self.targets = targets
        self.num_classes = num_classes

        self.threshold_examples_ids = set()
        
        self.is_ulb = is_ulb
        self.onehot = onehot

        self.transform = transform
        
        if self.is_ulb:
            if strong_transform is None:
                self.strong_transform = copy.deepcopy(transform)
                self.strong_transform.transforms.insert(0, RandAugment(3, 5))
        else:
            self.strong_transform = strong_transform

    def __getitem__(self, idx):
        if self.targets is None:
            target = -1
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        # set augmented images
        img = self.data[idx]
        if self.transform is None:
            return transforms.ToTensor()(img), target
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_w = self.transform(img)
            if not self.is_ulb:
                return idx, img_w, target
            else:
                if idx in self.threshold_examples_ids:
                    return idx, img_w, self.strong_transform(img), target, True
                else:
                    return idx, img_w, self.strong_transform(img), target, False

    def switch_threshold_examples(self, ids):
        self.threshold_examples_ids = set(ids)


    def __len__(self):
        return len(self.data)
