import os
from  jittor.dataset import Dataset
from PIL import Image


# 返回对应的(图片,GT图)列表
def make_dataset(root):
    image_path = os.path.join(root, 'image')
    mask_path = os.path.join(root, 'mask')
    img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.jpg')]
    return [(os.path.join(image_path, img_name + '.jpg'), os.path.join(mask_path, img_name + '.png')) for img_name in img_list]
class ImageSet(Dataset):
    def __init__(self, root, joint_transform=None, img_transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.total_len = len(self.imgs)
    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        # 同时对图像和GL做变换
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

