from torch.utils.data import Dataset
import torch
import os
from PIL import Image

class LicensePlateDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, "images")
        self.ann_dir = os.path.join(self.root_dir, "annotations")
        self.transforms = transforms
        self.images = sorted(os.listdir(self.img_dir))
        self.annotations = sorted(os.listdir(self.ann_dir))
        self.error_images = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.retrieve_image(idx)
        img_size = image.size
        if self.transforms is not None:
            image = self.transforms(image)
        target = self.retrieve_target(idx, img_size)
        return image, target

    def retrieve_image(self, idx):
        img_name = self.images[idx]
        image_path = os.path.join(self.img_dir, img_name)
        image = Image.open(image_path).convert("RGB")
        return image
    
    def retrieve_target(self, idx, img_size):
        ann_name = self.annotations[idx]
        ann_path = os.path.join(self.ann_dir, ann_name)
        boxes, labels, check_error = self.get_boxes_labels(ann_path, img_size)
        if check_error:
            self.error_images.append(ann_name)
        target = {"boxes": boxes, "labels": labels}
        return target
    
    def get_boxes_labels(self, path, img_size):
        with open(path, "r") as f:
            contents = f.readlines()
        labels = []
        boxes = []
        check_error = False
        img_w, img_h = img_size[0], img_size[1]
        for yolo_bbox in contents:
            yolo_bbox = yolo_bbox.strip().split()
            if len(yolo_bbox) < 5:
                check_error = True
                continue
            elif len(yolo_bbox) > 5:
                yolo_bbox = yolo_bbox[:5]
                check_error = True
            class_id, x_c, y_c, w, h = map(float, yolo_bbox)
            xmin = (x_c - w / 2) * img_w
            ymin = (y_c - h / 2) * img_h
            xmax = (x_c + w / 2) * img_w
            ymax = (y_c + h / 2) * img_h
            label = int(class_id) + 1
            labels.append(label)
            boxes.append([xmin, ymin, xmax, ymax])
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0, ), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        return boxes, labels, check_error