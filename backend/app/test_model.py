import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from classifier_head import HydrationClassifier


# load test embeddings 
data = torch.load("hydration_signal_embeddings.pt", map_location="cpu")

X_test = data["X_test"]
y_test = data["y_test"]


#load model checkpoint
ckpt = torch.load("hydration_best_model.pth", map_location="cpu")

input_dim   = ckpt["input_dim"]
num_classes = ckpt["num_classes"]

model = HydrationClassifier(input_dim, num_classes)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


inv_label_map = {
    0: "dehydrated",
    1: "hydrated",
    2: "mildly_dehydrated"
}

#test loop
correct = 0            
total   = 0
predictions = []

with torch.no_grad():
    for i in range(X_test.shape[0]):

        x = X_test[i].unsqueeze(0)  # [1,2304]
        y = y_test[i].item()

        logits = model(x)
        probs = F.softmax(logits, dim=1)

        pred_idx = probs.argmax(dim=1).item()
        conf     = probs.max().item()

        pred_lbl = inv_label_map[pred_idx]
        true_lbl = inv_label_map[y]

        correct += int(pred_idx == y)
        total   += 1

        predictions.append({
            "pred_label": pred_lbl,
            "true_label": true_lbl,
            "confidence": float(conf)
        })


# Final test accuracy 
test_acc = correct / total
print("\n=============================")
print(f"TEST ACCURACY: {test_acc:.4f}")
print("=============================\n")
