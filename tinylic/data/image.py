from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
class ImageFolder(Dataset):
    def __init__(self, roots, transform=None):
        im_exts = {'.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG'}
        self.samples = []

        # for current_dir in roots:
        #     current_dir = Path(current_dir)
        #     if not current_dir.is_dir():
        #         raise RuntimeError(f'Invalid directory "{current_dir}"')

        #     self.samples.extend(
        #         sorted(str(p) for p in current_dir.rglob("*") if p.suffix in im_exts)
        #     )
            
        splitdir = Path(roots)
        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)