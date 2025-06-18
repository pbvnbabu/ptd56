import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model import StudentNet
from train import train_model

df = pd.read_csv("student_data.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, stratify=y, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

model = StudentNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

train_model(model, train_loader, val_loader, optimizer, criterion)