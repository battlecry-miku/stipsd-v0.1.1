"""CPU training, followed by frozen-model (Strategy B) evaluation."""
import csv
import random
import numpy as np
import torch
from .resnet_model import create_model
from .data import normalize_targets, inverse_targets


def train(x, y, split, scaler, epochs, seed, output):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tx = torch.from_numpy(x[:, None])
    ty = torch.from_numpy(normalize_targets(y, scaler))
    rng = np.random.default_rng(seed)
    best, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(split["train"])
        loss_sum = 0.0
        for start in range(0, len(order), 16):
            batch = order[start:start+16]
            optimizer.zero_grad()
            loss = torch.nn.functional.smooth_l1_loss(model(tx[batch]), ty[batch])
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            val = torch.nn.functional.smooth_l1_loss(model(tx[split["val"]]), ty[split["val"]]).item()
        if not np.isfinite(val):
            raise FloatingPointError("Non-finite validation loss")
        if val < best:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "train_huber": loss_sum / len(order), "val_huber": val})
    model.load_state_dict(best_state)
    model.eval()
    torch.save({"model_state_dict": best_state, "scaler": scaler, "best_epoch": best_epoch,
                "seed": seed}, output / "checkpoint.pt")
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return model, best_epoch


def predict(model, image, scaler):
    with torch.no_grad():
        scaled = model(torch.from_numpy(image[None, None])).numpy()[0]
    return inverse_targets(scaled, scaler)
