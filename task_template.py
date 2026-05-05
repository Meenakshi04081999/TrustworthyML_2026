import os
import sys
import torch
import pandas as pd
import requests
import random
import argparse

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.models import resnet18
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# config
BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
OUTPUT_CSV = BASE / "submission.csv"

BASE_URL = "http://34.63.153.158"  # DONOT CHANGE
API_KEY = "c624f3b8d663751fcf05c23893ab116a"
TASK_ID = "01-mia"  # DONOT CHANGE


# dataset classes
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


# load datasets
print("Loading datasets...")
pub_ds = torch.load(PUB_PATH, weights_only=False)
priv_ds = torch.load(PRIV_PATH, weights_only=False)

# normalization (same as training)
MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Normalize(mean=MEAN, std=STD),
])

pub_ds.transform = transform
priv_ds.transform = transform

# load model
print("Loading model...")
model = resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Linear(512, 9)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()


def extract_features(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x


# testing

print("Building member feature bank...")

member_features = []

with torch.no_grad():
    for i in range(len(pub_ds)):
        sample = pub_ds[i]
        if sample is None:
            continue

        id_, img, label, membership = sample

        if membership != 1:
            continue  # only members

        img = img.to(device).unsqueeze(0)
        feat = extract_features(model, img)

        member_features.append(feat.cpu())

member_features = torch.cat(member_features, dim=0)


# feature membership approach
def compute_mia_scores(dataset):
    all_ids = []
    all_scores = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]

            if sample is None:
                continue

            id_, img, label = sample[:3]

            if img is None:
                continue

            img = img.to(device).unsqueeze(0)

            feat = extract_features(model, img).cpu()
            feat = F.normalize(feat, dim=1)

            # compute distance to all member features
            similarity = torch.matmul(member_features, feat.T).squeeze()

            k = 20
            topk = torch.topk(similarity, k).values

            logits = model(img)
            loss = F.cross_entropy(logits, torch.tensor([label]).to(device))

            score = topk.mean().item() - loss.item()

            all_ids.append(id_)
            all_scores.append(score)

    return all_ids, all_scores


print("computing MIA scores")

## Validation on public dataset
"""
print("Validating on public dataset...")

# compute scores on public dataset
pub_ids, pub_scores = compute_mia_scores(pub_ds)

# separate members and non-members
members = []
non_members = []

for i in range(len(pub_ds)):
    if pub_ds.membership[i] == 1:
        members.append(pub_scores[i])
    else:
        non_members.append(pub_scores[i])

# compute statistics
import numpy as np

m_mean = np.mean(members)
nm_mean = np.mean(non_members)

print("Member mean score:     ", m_mean)
print("Non-member mean score:", nm_mean)
print("Gap:", abs(m_mean - nm_mean))

# plot distributions
import matplotlib.pyplot as plt

plt.hist(members, bins=50, alpha=0.5, label="members")
plt.hist(non_members, bins=50, alpha=0.5, label="non-members")
plt.legend()
plt.title("MIA Score Distribution")
plt.savefig("distribution.png")

print("Saved distribution plot as distribution.png")
"""

ids, scores = compute_mia_scores(priv_ds)

df = pd.DataFrame({
    "id": ids,
    "score": scores
})
# df["score"] = df["score"].clip(0,1)


# scores
df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)

# create random submission (remove this later or it will rewrite your actual submission)
"""print("Creating random submission...")
ids = [str(i) for i in priv_ds.ids]

df = pd.DataFrame({
    "id": ids,
    "score": [random.random() for _ in ids]
})

df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)"""


# submit


def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


parser = argparse.ArgumentParser(description="Submit a CSV file to the server.")
args = parser.parse_args()

submit_path = OUTPUT_CSV

if not submit_path.exists():
    die(f"File not found: {submit_path}")

try:
    with open(submit_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files={"file": (submit_path.name, f, "application/csv")},
            timeout=(10, 600),
        )
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    if resp.status_code == 413:
        die("Upload rejected: file too large (HTTP 413).")

    resp.raise_for_status()

    print("Successfully submitted.")
    print("Server response:", body)
    submission_id = body.get("submission_id")
    if submission_id:
        print(f"Submission ID: {submission_id}")

except requests.exceptions.RequestException as e:
    detail = getattr(e, "response", None)
    print(f"Submission error: {e}")
    if detail is not None:
        try:
            print("Server response:", detail.json())
        except Exception:
            print("Server response (text):", detail.text)
    sys.exit(1)



