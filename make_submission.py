import os
import numpy as np
import pandas as pd
import torch, torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import timm
import joblib
from PIL import Image


CLASSIFIER_PATH = "./logistic_regression.pkl"
TEST_DATASET = "./data/test/"
SUBMISSION_PATH = "./data/submission.csv"
IMG_SIZE = 448


def set_requires_grad(model, value=False):
    for param in model.parameters():
        param.requires_grad = value


def init_model(device):
    model = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in1k', pretrained=True, num_classes=0)
    set_requires_grad(model, False)
    model = model.to(device)
    model = model.eval()
    return model


def load_classifier(path):
    clf = joblib.load(path)
    return clf


class ArtDataset(Dataset):
    def __init__(self, root_dir, csv_path=None, transform=None):
        self.transform = transform
        self.files = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]
        self.targets = None
        if csv_path:
            df = pd.read_csv(csv_path, sep="\t")
            self.targets = df["label_id"].tolist()
            self.files = [os.path.join(root_dir, fname) for fname in df["image_name"].tolist()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert('RGB')
        target = self.targets[idx] if self.targets else -1
        if self.transform:
            image = self.transform(image)
        return image, target


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = init_model(device)
    clf = load_classifier(CLASSIFIER_PATH)
    print("Model is loaded")

    trans = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])])

    dset = ArtDataset(TEST_DATASET, transform=trans)
    batch_size = 16
    num_workers = 0
    testloader = torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    all_image_names = [item.split("/")[-1] for item in dset.files]
    all_preds = []
    with torch.no_grad():
        for idx, (images, _) in enumerate(testloader, 0):
            images = images.to(device)
            outputs = model(images).cpu().numpy()
            preds = clf.predict(outputs)
            all_preds.extend(preds)

    print("Get predicts")

    with open(SUBMISSION_PATH, "w") as f:
        f.write("image_name\tlabel_id\n")
        for name, cl_id in zip(all_image_names, all_preds):
            f.write(f"{name}\t{cl_id}\n")
        print("Submission is written")
