import os
import torch
import torchvision
import pandas as pd
import albumentations as A
import matplotlib.image as im

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# parent directory

PARENT_DIR ="/home/ranjani.bala@zucisystems.com/workspace/dev/loris_gift/data/tablev1.0"


# output directory
OUTPUT_DIR = "/home/ranjani.bala@zucisystems.com/workspace/dev/loris_gift/data/tablev1.0/output"
OUTPUT_DIR = os.path.join(PARENT_DIR, OUTPUT_DIR)

# train directory
TRAIN_DIR = "train/"
TRAIN_FILES = os.path.join(PARENT_DIR, TRAIN_DIR)

# test directory
TEST_DIR = "test/"
TEST_FILES = os.path.join(PARENT_DIR, TEST_DIR)

# bounding box value file
BBOX_VALUE = "column_annotations.csv"
BBOX_FILE = os.path.join(PARENT_DIR, BBOX_VALUE)

MODEL_NAME = "table_attribution_col_v1.0.pth"

EPOCHS = 500
BATCH_SIZE = 2


class ImageDataset(Dataset):
    def __init__(self, root=PARENT_DIR, tt="train", transforms_tt=True):

        df = pd.read_csv(BBOX_FILE)

        self.root = root
        self.transforms = transforms
        self.imgs = df["image_id"].unique()
        self.df = df
        self.tt = tt
        self.transform = None
        if transforms_tt is True:
            self.transform = A.Compose([A.RandomBrightnessContrast(p=0.2)],
                                       bbox_params=A.BboxParams(format='coco',
                                                                label_fields=['class_labels']))

    def __getitem__(self, idx):

        image_id = self.imgs[idx]

        _image = im.imread("{}/{}/{}.jpg".format(PARENT_DIR, self.tt, image_id))

        records = self.df["bbox"][self.df['image_id'] == image_id].values
        boxes = []
        for i, l in enumerate(records):
            b = [float(num) for num in l[1:-1].split(",")]
            boxes.append(b)

        target = {}
        target["labels"] = torch.ones((records.shape[0],), dtype=torch.int64)

        target["iscrowd"] = torch.zeros((records.shape[0],), dtype=torch.int64)

        if self.transforms is not None:
            transformed = self.transform(image=_image, bboxes=boxes, class_labels=["table"] * len(boxes))
            img = transformed['image']
            img = torchvision.transforms.functional.to_tensor(img)
            transformed_bboxes = transformed['bboxes']
            bboxes = []
            for b in transformed_bboxes:
                bboxes.append([b[0], b[1], b[0] + b[2], b[1] + b[3]])
            target["boxes"] = torch.as_tensor(bboxes, dtype=torch.float32)

        if self.transform is None:
            bboxes = []
            for b in boxes:
                bboxes.append([b[0], b[1], b[0] + b[2], b[1] + b[3]])
            target["boxes"] = bboxes
            img = torchvision.transforms.functional.to_tensor(_image)

        del records
        del _image

        return img, target

    def __len__(self):
        return len(self.imgs)


class tableDetectionTrain:
    # The init method or constructor
    def __init__(self, parent_dir, train_dir, test_dir, bbox_file, epochs, batch_size, model_name, output_dir):
        # Instance Variable
        self.parent_dir = parent_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.bbox_file = bbox_file
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.model_name = model_name

    """  table detection train """

    def table_detection_train(self):

        train_df = pd.read_csv(self.bbox_file)

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True,
                                                                     pretrained_backbone=True)
        num_classes = 2
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        dataset = ImageDataset()
        data_loader = DataLoader(dataset, self.batch_size, collate_fn=lambda batch: list(zip(*batch)))
        model = model.to(device)
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.0008, momentum=0.9, weight_decay=0.0005)
        lossAvg, lossPer = [], []
        for epoch in range(self.epochs):
            total_loss, count = 0, 0
            for batch in data_loader:
                images, targets = batch

                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                if count % 10 == 0:
                    print("loss: {}".format(losses.item()))
                    lossPer.append(losses.item())
                count += 1
                total_loss += losses.item()
            g = total_loss / count
            lossAvg.append(g)
            print("END EPOCH #{} avg: {}".format(epoch, total_loss / count))

            torch.cuda.empty_cache()

        torch.save(model, self.model_name)

        torch.cuda.empty_cache()

        return self.model_name


table_detection_train = tableDetectionTrain(PARENT_DIR, TRAIN_FILES, TEST_FILES, BBOX_FILE, EPOCHS, BATCH_SIZE,
                                                  MODEL_NAME, OUTPUT_DIR)
table_detection_train.table_detection_train()
