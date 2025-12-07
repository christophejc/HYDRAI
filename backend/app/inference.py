import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from NormWear.main_model import NormWearModel
from .classifier_head import HydrationClassifier

WINDOW = 150                    # expected time steps
SAMPLING_RATE = 50             # Hz

LABELS = {
    0: "dehydrated",
    1: "hydrated",
    2: "mildly_dehydrated"
}
CURRENT_DIR = Path(__file__).parent

MODEL_PATH = CURRENT_DIR / "hydration_best_model.pth"
ENCODER_WEIGHTS = CURRENT_DIR / "normwear_last_checkpoint-15470-correct.pth"
#ENCODER_WEIGHTS = CURRENT_DIR / "../../ml_model/normwear_last_checkpoint-15470-correct.pth"

device = torch.device("cpu")

encoder = NormWearModel(
    weight_path=ENCODER_WEIGHTS.as_posix(),
    optimized_cwt=True
).to(device)
encoder.eval()
encoder.requires_grad_(False)

 
ckpt = torch.load(MODEL_PATH, map_location=device)

model = HydrationClassifier(
    input_dim=ckpt["input_dim"],
    num_classes=ckpt["num_classes"]
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


def classify_window_single(window_150x3: np.ndarray):
    """
    Classify a SINGLE 150x3 window.

    Input:
        window_150x3 : numpy array of shape (150, 3)
                       columns: [gsr_raw, temp_raw, hr_raw]

    Returns
    -------
    str
        One of: "dehydrated", "hydrated", "mildly_dehydrated"
    """

    if window_150x3.shape != (150, 3):
        raise ValueError(
            f"Expected window shape (150,3), got {window_150x3.shape}"
        )

    # Convert to tensor [1,3,150]
    sensor_tensor = (
        torch.tensor(window_150x3, dtype=torch.float32)
        .permute(1, 0)      # [3,150]
        .unsqueeze(0)       # [1,3,150]
    )

    # embed using normwear
    with torch.no_grad():
        embs = encoder.get_embedding(
            sensor_tensor,
            sampling_rate=SAMPLING_RATE
        )

        # patch pooling
        if embs.dim() == 4:
            x = embs.mean(dim=2).flatten(start_dim=1)
        else:
            raise ValueError(
                f"Unexpected embedding shape {embs.shape}"
            )

    #predict sample
    with torch.no_grad():
        logits = model(x)
        pred_idx = logits.argmax(dim=1).item()

    # Return only the label
    return LABELS[pred_idx]
