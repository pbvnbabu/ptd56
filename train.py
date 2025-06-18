import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=10):
    writer = SummaryWriter()
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        correct = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                y_true.extend(targets.tolist())
                y_pred.extend(predicted.tolist())

        val_loss = total_val_loss / len(val_loader)
        val_acc = correct / len(val_loader.dataset)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Acc/Val", val_acc, epoch)

        print(f"[{epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    writer.close()
    torch.save(model.state_dict(), "student_model.pth")
    print("✅ Model saved to student_model.pth")