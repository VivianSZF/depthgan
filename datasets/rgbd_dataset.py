# python3.7
"""Contains the class of dataset."""

import os
import zipfile
import numpy as np
import cv2
import json
import PIL

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


try:
    import turbojpeg
    BASE_DIR = os.path.dirname(os.path.relpath(__file__))
    LIBRARY_NAME = 'libturbojpeg.so.0'
    LIBRARY_PATH = os.path.join(BASE_DIR, LIBRARY_NAME)
    jpeg = turbojpeg.TurboJPEG(LIBRARY_PATH)
except ImportError:
    jpeg = None

__all__ = ['RGBDDataset']

_FORMATS_ALLOWED = ['dir', 'lmdb', 'list', 'zip']


def crop_resize_image(image, size):
    """Crops a square patch and then resizes it to the given size.

    Args:
        image: The input image to crop and resize.
        size: An integer, indicating the target size.

    Returns:
        An image with target size.

    Raises:
        TypeError: If the input `image` is not with type `numpy.ndarray`.
        ValueError: If the input `image` is not with shape [H, W, C].
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f'Input image should be with type `numpy.ndarray`, '
                        f'but `{type(image)}` is received!')
    if image.ndim != 3:
        raise ValueError(f'Input image should be with shape [H, W, C], '
                         f'but `{image.shape}` is received!')

    height, width, channel = image.shape
    short_side = min(height, width)
    image = image[(height - short_side) // 2:(height + short_side) // 2,
                  (width - short_side) // 2:(width + short_side) // 2]
    if channel == 3:
        pil_image = PIL.Image.fromarray(image)
        pil_image = pil_image.resize((size, size), PIL.Image.ANTIALIAS)
        image = np.asarray(pil_image)
    elif channel == 1:
        image = cv2.resize(image, (size, size))
        if image.ndim == 2:
            image = image[:,:,np.newaxis]
    assert image.shape == (size, size, channel)
    return image


def progressive_resize_image(image, size):
    """Resizes image to target size progressively.

    Different from normal resize, this function will reduce the image size
    progressively. In each step, the maximum reduce factor is 2.

    NOTE: This function can only handle square images, and can only be used for
    downsampling.

    Args:
        image: The input (square) image to resize.
        size: An integer, indicating the target size.

    Returns:
        An image with target size.

    Raises:
        TypeError: If the input `image` is not with type `numpy.ndarray`.
        ValueError: If the input `image` is not with shape [H, W, C].
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f'Input image should be with type `numpy.ndarray`, '
                        f'but `{type(image)}` is received!')
    if image.ndim != 3:
        raise ValueError(f'Input image should be with shape [H, W, C], '
                         f'but `{image.shape}` is received!')

    height, width, channel = image.shape
    assert height == width
    assert height >= size
    num_iters = int(np.log2(height) - np.log2(size))
    for _ in range(num_iters):
        height = max(height // 2, size)
        image = cv2.resize(image, (height, height),
                           interpolation=cv2.INTER_LINEAR)
    assert image.shape == (size, size, channel)
    return image


def resize_image(image, size):
    """Resizes image to target size.

    NOTE: We use adaptive average pooing for image resizing. Instead of bilinear
    interpolation, average pooling is able to acquire information from more
    pixels, such that the resized results can be with higher quality.

    Args:
        image: The input image tensor, with shape [C, H, W], to resize.
        size: An integer or a tuple of integer, indicating the target size.

    Returns:
        An image tensor with target size.

    Raises:
        TypeError: If the input `image` is not with type `torch.Tensor`.
        ValueError: If the input `image` is not with shape [C, H, W].
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f'Input image should be with type `torch.Tensor`, '
                        f'but `{type(image)}` is received!')
    if image.ndim != 3:
        raise ValueError(f'Input image should be with shape [C, H, W], '
                         f'but `{image.shape}` is received!')

    image = F.adaptive_avg_pool2d(image.unsqueeze(0), size).squeeze(0)
    return image


def normalize_image(image, mean=127.5, std=127.5):
    """Normalizes image by subtracting mean and dividing std.

    Args:
        image: The input image tensor to normalize.
        mean: The mean value to subtract from the input tensor. (default: 127.5)
        std: The standard deviation to normalize the input tensor. (default:
            127.5)

    Returns:
        A normalized image tensor.

    Raises:
        TypeError: If the input `image` is not with type `torch.Tensor`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f'Input image should be with type `torch.Tensor`, '
                        f'but `{type(image)}` is received!')
    out = (image - mean) / std
    return out


class ZipLoader(object):
    """Defines a class to load zip file.

    This is a static class, which is used to solve the problem that different
    data workers can not share the same memory.
    """
    files = dict()

    @staticmethod
    def get_zipfile(file_path):
        """Fetches a zip file."""
        zip_files = ZipLoader.files
        if file_path not in zip_files:
            zip_files[file_path] = zipfile.ZipFile(file_path, 'r')
        return zip_files[file_path]

    @staticmethod
    def get_image(file_path, image_path, depth=False):
        """Decodes an image from a particular zip file."""
        zip_file = ZipLoader.get_zipfile(file_path)
        image_str = zip_file.read(image_path)
        if depth:
            image_np = np.frombuffer(image_str, np.uint8) # uint8-encode
            image = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED) # TBC
        else:
            image_np = np.frombuffer(image_str, np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        return image

    @staticmethod
    def get_image_paths(file_path, dataset_json_for_zip=None):
        zip_file = ZipLoader.get_zipfile(file_path)
        if dataset_json_for_zip is not None:
            with open(dataset_json_for_zip, 'r') as jsonfile:
                paths = json.load(jsonfile)
                rgb_paths = sorted(paths['rgb_paths'])
                depth_paths = sorted(paths['depth_paths'])
                paths_file = {'rgb':rgb_paths,
                              'depth':depth_paths}
        elif 'dataset.json' in zip_file.namelist():
            with zip_file.open('dataset.json', 'r') as jsonfile:
                paths = json.load(jsonfile)
                rgb_paths = sorted(paths['rgb_paths'])
                depth_paths = sorted(paths['depth_paths'])
                paths_file = {'rgb':rgb_paths,
                              'depth':depth_paths}
        else:
            rgb_paths = [f for f in zip_file.namelist()
                           if ('.jpg' in f or '.jpeg' in f)]
            depth_paths = [f for f in zip_file.namelist()
                           if ('.png' in f)]
            paths_file = {'rgb':sorted(rgb_paths),
                          'depth':sorted(depth_paths)}
        return paths_file


class RGBDDataset(Dataset):
    """Defines the base dataset class.

    This class supports loading data from a full-of-image folder, a lmdb
    database, or an image list. Images will be pre-processed based on the given
    `transform` function before fed into the data loader.

    NOTE: The loaded data will be returned as a directory, where there must be
    a key `image`.
    """
    def __init__(self,
                 root_dir,
                 resolution,
                 file_format='zip',
                 image_list_path=None,
                 annotation_path=None,
                 mirror=0.0,
                 progressive_resize=True,
                 crop_resize_resolution=-1,
                 transform=normalize_image,
                 transform_kwargs=None,
                 img_channel=4,
                 **_unused_kwargs):
        """Initializes the dataset.

        Args:
            root_dir: Root directory containing the dataset.
            resolution: The resolution of the returned image.
            file_format: Format the dataset is stored. Supports `dir`, `lmdb`,
                and `list`. (default: `dir`)
            image_list_path: Path to the image list. This field is required if
                `file_format` is `list`. (default: None)
            mirror: The probability to do mirror augmentation. (default: 0.0)
            progressive_resize: Whether to resize images progressively.
                (default: True)
            crop_resize_resolution: The resolution of the output after crop
                and resize. (default: -1)
            transform: The transform function for pre-processing.
                (default: `datasets.transforms.normalize_image()`)
            transform_kwargs: The additional arguments for the `transform`
                function. (default: None)

        Raises:
            ValueError: If the input `file_format` is not supported.
            NotImplementedError: If the input `file_format` is not implemented.
        """
        if file_format.lower() not in _FORMATS_ALLOWED:
            raise ValueError(f'Invalid data format `{file_format}`!\n'
                             f'Supported formats: {_FORMATS_ALLOWED}.')

        self.root_dir = root_dir
        self.resolution = resolution
        self.file_format = file_format.lower()
        self.image_list_path = image_list_path
        self.annotation_path = annotation_path
        self.mirror = np.clip(mirror, 0.0, 1.0)
        self.progressive_resize = progressive_resize
        self.crop_resize_resolution = crop_resize_resolution
        self.transform = transform
        self.transform_kwargs = transform_kwargs or dict()
        self.img_channel = img_channel

        if self.file_format == 'dir':
            self.rgb_paths = sorted(os.listdir(os.path.join(self.root_dir, 'rgb')))
            self.depth_paths = sorted(os.listdir(os.path.join(self.root_dir, 'depth')))
            self.num_samples = len(self.rgb_paths)
        elif self.file_format == 'list':
            self.metas = []
            assert os.path.isfile(self.image_list_path)
            with open(self.image_list_path) as f:
                for line in f:
                    fields = line.rstrip().split(' ')
                    if len(fields) == 1:
                        self.metas.append((fields[0], None))
                    else:
                        assert len(fields) == 2
                        self.metas.append((fields[0], int(fields[1])))
            self.num_samples = len(self.metas)
        elif self.file_format == 'zip':
            image_paths = ZipLoader.get_image_paths(self.root_dir, annotation_path)
            self.rgb_paths = image_paths['rgb']
            self.depth_paths = image_paths['depth']
            self.num_samples = len(self.rgb_paths)
            assert len(self.rgb_paths) == len(self.depth_paths)
        else:
            raise NotImplementedError(f'Not implemented data format '
                                      f'`{self.file_format}`!')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = dict()

        # Load data.
        if self.file_format == 'dir':
            rgb_path = self.rgb_paths[idx]
            depth_path = self.depth_paths[idx]
            rgb_image = cv2.imread(os.path.join(self.root_dir, 'rgb', rgb_path))
            depth_image = cv2.imread(os.path.join(self.root_dir, 'depth', depth_path), cv2.IMREAD_UNCHANGED)
        elif self.file_format == 'zip':
            rgb_path = self.rgb_paths[idx]
            depth_path = self.depth_paths[idx]
            rgb_image = ZipLoader.get_image(self.root_dir, rgb_path, depth=False)
            depth_image = ZipLoader.get_image(self.root_dir, depth_path, depth=True)
        else:
            raise NotImplementedError(f'Not implemented data format '
                                      f'`{self.file_format}`!')

        rgb_image = rgb_image[:, :, ::-1]  # Converts BGR (cv2) to RGB.
        if depth_image.ndim == 2:
            depth_image = depth_image[:, :, np.newaxis]

        # Transform image.
        if self.crop_resize_resolution > 0:
            rgb_image = crop_resize_image(rgb_image, self.crop_resize_resolution)
            depth_image = crop_resize_image(depth_image, self.crop_resize_resolution)
        if self.progressive_resize:
            rgb_image = progressive_resize_image(rgb_image, self.resolution)
            depth_image = progressive_resize_image(depth_image, self.resolution)
        rgb_image = rgb_image.transpose(2, 0, 1).astype(np.float32)
        depth_image = depth_image.transpose(2, 0, 1).astype(np.float32)
        if np.random.uniform() < self.mirror:
            rgb_image = rgb_image[:, :, ::-1]  # CHW
        rgb_image = torch.FloatTensor(rgb_image.copy())
        depth_image = torch.FloatTensor(depth_image.copy())
        if not self.progressive_resize:
            rgb_image = resize_image(rgb_image, self.resolution)
            depth_image = resize_image(depth_image, self.resolution)

        if self.transform is not None:
            rgb_image = self.transform(rgb_image, **self.transform_kwargs)
            depth_image = self.transform(depth_image, mean=30000.0, std=30000.0)
        assert depth_image.shape[0] == 1
        assert rgb_image.shape[0] == 3
        if self.img_channel == 4:
            image = torch.cat((rgb_image, depth_image), dim=0)
            assert image.shape[0] == 4
        else:
            image = rgb_image
        data.update({'image': image})

        return data

    def save_items(self, save_dir, tag=None):
        """Saves the item list to disk.

        Name of the saved file is set as `${self.dataset_name}_item_list.txt`

        Args:
            save_dir: The directory under which to save the item list.
        """
        return
    
    def info(self):
        """Collects the information of the dataset.

        Please append new information in derived class if needed.
        """
        dataset_info = {
            'Type': 'RGBDDataset',
            'Root dir': self.root_dir,
            'Dataset file format': self.file_format,
            'Annotation path': self.annotation_path,
            'Num samples in dataset': self.num_samples,
            'Mirror': self.mirror,
        }
        return dataset_info



