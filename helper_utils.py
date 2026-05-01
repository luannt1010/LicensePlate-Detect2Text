import os
import torch
from tqdm.auto import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from dataset import LicensePlateDataset
from torch.utils.data import random_split, DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_pil_image


def draw_bbox(image, target, class_map):
    boxes = target["boxes"]
    tar_labels = target["labels"]
    labels = [class_map[int(i)] for i in tar_labels]
    result = vutils.draw_bounding_boxes(image=image, 
                                    boxes=boxes, 
                                    labels=labels,           
                                    colors="red",  
                                    width=4, font_size=30                  
                                   )
    result = to_pil_image(result)
    result.show()

def split_dataset(datasets, val_factor, test_factor):
    total_size = len(datasets)
    val_size = int(total_size * val_factor)
    test_size = int(total_size * test_factor)
    train_size = total_size - (val_size + test_size)
    train_dataset, val_dataset, test_dataset = random_split(datasets, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset

def create_dataset_splits(root_dir, val_factor, test_factor, transform=None):
    dataset = LicensePlateDataset(root_dir, transform)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, val_factor, test_factor)
    return dataset, train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=lambda x: list(zip(*x)))
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, collate_fn=lambda x: list(zip(*x)))
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, collate_fn=lambda x: list(zip(*x)))
    return train_loader, val_loader, test_loader

def load_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def training_loop(model, train_loader, val_loader, optimizer, num_epochs, save_path, device='cpu'):
    history = {"training_loss": [], "map50": [], "map": []}
    best_map_all = 0.0
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox').to(device)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False)
        for images, targets in train_pbar:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            loss_dict = model(images, targets) # {loss1: tensor1, loss2:tensor2,..}
            losses = sum(loss.to(device) for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item() * len(images)
        avg_loss = epoch_loss / len(train_loader.dataset)
        history["training_loss"].append(avg_loss)
        model.eval()
        metric.reset()
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", leave=False)
        with torch.no_grad():
            for images, targets in val_pbar:
                images = [image.to(device) for image in images]
                outputs = model(images)
                preds, gts = [], []
                for i in range(len(images)):
                    preds.append({"boxes": outputs[i]["boxes"].cpu(),
                                  "scores": outputs[i]["scores"].cpu(),
                                  "labels": outputs[i]["labels"].cpu()})
                    gts.append({"boxes": targets[i]["boxes"].cpu(), 
                                "labels": targets[i]["labels"].cpu()})
                metric.update(preds, gts)
        results = metric.compute()
        map50 = results["map_50"].item()
        map_all = results["map"].item()
        history["map50"].append(map50)
        history["map"].append(map_all)
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}   |   mAP@0.5={map50:.4f}   |   mAP={map_all:.4f}")
        os.makedirs(save_path, exist_ok=True)
        if map_all > best_map_all:
            best_map_all = map_all
            if save_path:
                torch.save(model.state_dict(), os.path.join(save_path, "best.pth"))
        if epoch + 1 == num_epochs:
            if save_path:
                torch.save(model.state_dict(), os.path.join(save_path, "last.pth"))
    print(f"Model is saved at folder {save_path}.")
    return model, history

def visualize_metrics(metrics):
    train_loss = metrics["training_loss"]
    nums_epochs = range(1, len(train_loss)+1)
    map50 = metrics["map50"]
    mAP = metrics["map"]
    _, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].plot(nums_epochs, train_loss, linewidth=3, color="red")
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")

    ax[1].plot(nums_epochs, map50, linewidth=3, color="red")
    ax[1].set_title("mAP@0.5")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("mAP")

    ax[2].plot(nums_epochs, mAP, linewidth=3, color="red")
    ax[2].set_title("mAP for all")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("mAP")
    plt.tight_layout()
    plt.show()

def predict(img, model, class_map):
    model.eval()
    with torch.no_grad():
        predictions = model(img)
    labels = [class_map[int(i)] for i in predictions["labels"]]
    result = vutils.draw_bounding_boxes(img, boxes=predictions["boxes"],
                                     labels=labels,
                                     colors="red",
                                     width=4, font_size=30)
    img = to_pil_image(result.detach())
    img.show()


        

                
