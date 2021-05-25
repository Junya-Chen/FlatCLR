from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
import os


class ContrastiveLearningDataset:
    def __init__(self, root):
        self.root = root

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1, gaussian=True):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        if gaussian:
            data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.RandomApply([color_jitter], p=0.8),
                                                  transforms.RandomGrayscale(p=0.2),
                                                  GaussianBlur(kernel_size=int(0.1 * size)),
                                                  transforms.ToTensor()])
        else:
            data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.RandomApply([color_jitter], p=0.8),
                                                  transforms.RandomGrayscale(p=0.2),
                                                  transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views, gaussian=False):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(32, s=.5, gaussian=False),
                                                              n_views),
                                                              download=True),
                          
                         'cifar100': lambda: datasets.CIFAR100(self.root, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(32, s=.5, gaussian=False),
                                                              n_views),
                                                              download=True),
                          
                          'imagenet': lambda: datasets.ImageFolder(os.path.join(self.root, 'imagenet/train'),
                                                           transform=ContrastiveLearningViewGenerator(
                                                           self.get_simclr_pipeline_transform(224, s=1, gaussian=True),
                                                           n_views)),

                          'stl10': lambda: datasets.STL10(self.root, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()