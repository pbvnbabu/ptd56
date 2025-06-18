import torch
import csv
from datetime import datetime

def log_prediction(features, label, confidence):
    row = {
        "timestamp": datetime.now().isoformat(),
        "hours_studied": features[0],
        "attendance": features[1],
        "participation": features[2],
        "previous_score": features[3],
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    }
    with open("prediction_log.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)

def predict(model, student_features):
    model.eval()
    input_tensor = torch.tensor([student_features], dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence = torch.max(probabilities).item()
        prediction = torch.argmax(output, dim=1).item()
    label = "PASS" if prediction == 1 else "FAIL"
    log_prediction(student_features, label, confidence)
    return label, confidence