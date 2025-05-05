import os
import h5py
import warnings
from PIL import Image
from typing import Any, Callable, Optional, Tuple

from .extended import ExtendedVisionDataset

H5_IMAGE_KEY= 'image'

class WSIDataset(ExtendedVisionDataset):
    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        super().__init__(root, transforms, transform, target_transform)

        self.folder_path = root
        self.dataset_name = os.path.splitext(os.path.basename(root))[0]
        with h5py.File(self.folder_path, 'r') as h5file:
            self.length = h5file[H5_IMAGE_KEY].shape[0]

    def _open_hdf5(self) -> None:
        self._h5file = h5py.File(self.folder_path, 'r')
        self._dataset = self._h5file[H5_IMAGE_KEY]
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if not hasattr(self, "_h5file"):
            self._open_hdf5()
        image = Image.fromarray(self._dataset[index])
        target = self.dataset_name

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target 

    def __len__(self) -> int:
        return self.length
    

