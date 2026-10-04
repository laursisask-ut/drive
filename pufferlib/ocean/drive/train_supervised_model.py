import random
import os

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
from tqdm import tqdm
import wandb


from pufferlib.ocean.torch import Wayformer

class PregeneratedDataset(Dataset):
    def __init__(self, archive_path):
        super().__init__()
        with np.load(archive_path) as data:
            inputs = data["inputs"]
            if inputs.ndim != 3 or inputs.shape[-1] != 1936:
                raise ValueError("Wrong input shape: regenerate supervised_data.npz with create_supervised_data.py")
            self.inputs = torch.from_numpy(inputs).to(torch.float32)
            # Hide reward preferences and all history in both splits, keeping the full shape.
            self.inputs[..., :7] = 0
            self.inputs[..., 20:95] = 0
            self.actions = torch.from_numpy(data["actions"]).to(torch.int64)
            self.waypoints = torch.from_numpy(data["waypoints"]).to(torch.float32)
            self.waypoint_masks = torch.from_numpy(data["waypoint_masks"]).to(torch.bool)
        assert self.inputs.shape[:2] == self.actions.shape[:2]

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, index):
        return self.inputs[index], self.actions[index], self.waypoints[index], self.waypoint_masks[index]

class FlattenedDataset(Dataset):
    def __init__(self, nested_dataset):
        super().__init__()
        self.nested_dataset = nested_dataset
        self.steps = nested_dataset[0][0].shape[0] if len(nested_dataset) else 0

    def __len__(self):
        return len(self.nested_dataset) * self.steps

    def __getitem__(self, index):
        inputs, actions, waypoints, waypoint_mask = self.nested_dataset[index // self.steps]
        return inputs[index % self.steps], actions[index % self.steps], waypoints[index % self.steps], waypoint_mask[index % self.steps]

def main():
    torch.manual_seed(2027)
    torch.set_float32_matmul_precision("high")
    random.seed(2026)

    model = Wayformer(ego_dim=95, action_dim=[7, 13])
    #model.load_state_dict(torch.load("/home/laur/Dev/PufferDrive/experiments/puffer_drive_n1uaeele/model_puffer_drive_003577.pt"))

    # Zero-input weights stay zero during training, so deployment ignores these
    # features too. Preferences enter through both a token and FiLM conditioning.
    history_columns = [
        i for i, obs_index in enumerate(model.encoder.ego_state_indices)
        if 20 <= obs_index < 95
    ]
    with torch.no_grad():
        model.encoder.ego_state_proj[0].weight[:, history_columns] = 0
        model.encoder.reward_pref_proj[0].weight.zero_()
        model.encoder.reward_film[0].weight.zero_()

    epochs = int(os.environ.get("EPOCHS", "10"))
    batch_size = int(os.environ.get("BATCH_SIZE", 1024))

    wandb.login()
    wandb.init(project="drive-supervised", config={"epochs": epochs, "batch_size": batch_size})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    original_model = model
    model = torch.compile(model)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    dataset = PregeneratedDataset("supervised_data.npz")
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])

    train_loader = DataLoader(FlattenedDataset(train_dataset), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(FlattenedDataset(val_dataset), batch_size=batch_size)

    for epoch in range(epochs):
        print(f"starting epoch {epoch}")

        train_loss_sum = 0
        train_accel_accuracy_sum = 0
        train_steering_accuracy_sum = 0
        train_updates = 0
        bar = tqdm(train_loader)
        for inputs, target_actions, _waypoints, _waypoint_masks in bar:
            inputs = inputs.to(device)
            target_actions = target_actions.to(device)
            model.train()

            optimizer.zero_grad()
            (predicted_accel, predicted_steering), _value = model(inputs)
            accel_loss = F.cross_entropy(predicted_accel, target_actions[:, 0])
            steering_loss = F.cross_entropy(predicted_steering, target_actions[:, 1])
            loss = accel_loss + steering_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            wandb.log({
                "batch_accel_loss": accel_loss.item(),
                "batch_steering_loss": steering_loss.item(),
                "batch_loss": loss.item(),
                "grad_norm": grad_norm.item()
            })
            optimizer.step()
            train_loss_sum += loss.item()

            with torch.no_grad():
                train_accel_accuracy_sum += (torch.argmax(predicted_accel, dim=1) == target_actions[:, 0]).float().mean()
                train_steering_accuracy_sum += (torch.argmax(predicted_steering, dim=1) == target_actions[:, 1]).float().mean()

            train_updates += 1
            bar.set_description(f"Train loss: {train_loss_sum / train_updates:.5f}, accel accuracy: {train_accel_accuracy_sum / train_updates:.5f}, steering accuracy: {train_steering_accuracy_sum / train_updates:.5f}")

        with torch.no_grad():
            model.eval()
            loss_sum = 0
            val_accel_accuracy_sum = 0
            val_steering_accuracy_sum = 0
            accel_preds = []
            accel_y_true = []
            steering_preds = []
            steering_y_true = []
            for inputs, target_actions, _waypoints, _waypoint_masks in val_loader:
                inputs = inputs.to(device)
                target_actions = target_actions.to(device)
                (predicted_accel, predicted_steering), _value = model(inputs)
                loss_sum += F.cross_entropy(predicted_accel, target_actions[:, 0])
                loss_sum += F.cross_entropy(predicted_steering, target_actions[:, 1])
                val_accel_accuracy_sum += (torch.argmax(predicted_accel, dim=1) == target_actions[:, 0]).float().mean()
                val_steering_accuracy_sum += (torch.argmax(predicted_steering, dim=1) == target_actions[:, 1]).float().mean()

                accel_preds += torch.argmax(predicted_accel, dim=1).tolist()
                accel_y_true += target_actions[:, 0].tolist()

                steering_preds += torch.argmax(predicted_steering, dim=1).tolist()
                steering_y_true += target_actions[:, 1].tolist()

            print(f"Train loss: {train_loss_sum / train_updates:.5f}, validation loss: {loss_sum / len(val_loader):.5f}, "
                    f"train accel accuracy: {train_accel_accuracy_sum / train_updates:.4f}, " 
                    f"validation accel accuracy: {val_accel_accuracy_sum / len(val_loader):4f}, "
                    f"train steering accuracy: {train_steering_accuracy_sum / train_updates:.4f}, "
                    f"validation steering accuracy: {val_steering_accuracy_sum / len(val_loader):.4f}")

            accel_confusion_matrix = wandb.plot.confusion_matrix(
                preds=accel_preds,
                y_true=accel_y_true,
                class_names=[-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0],
                title="Acceleration Confusion Matrix"
            )

            steering_confusion_matrix = wandb.plot.confusion_matrix(
                preds=steering_preds,
                y_true=steering_y_true,
                class_names=[-0.667, -0.5, -0.333, -0.167, -0.08, -0.01, 0.0, 0.01, 0.08, 0.167, 0.333, 0.500, 0.667],
                title="Steering Confusion Matrix"
            )

            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss_sum / train_updates,
                "validation_loss": (loss_sum / len(val_loader)).item(),
                "train_accel_accuracy": train_accel_accuracy_sum / train_updates,
                "validation_accel_accuracy": val_accel_accuracy_sum / len(val_loader),
                "train_steering_accuracy": train_steering_accuracy_sum / train_updates,
                "validation_steering_accuracy": val_steering_accuracy_sum / len(val_loader),
                "learning_rate": scheduler.get_last_lr()[0],
                "acceleration_confusion_matrix": accel_confusion_matrix,
                "steering_confusion_matrix": steering_confusion_matrix
            })

        scheduler.step()
        torch.save(original_model.state_dict(), "supervised_wayformer.pt")


if __name__ == "__main__":
    main()
