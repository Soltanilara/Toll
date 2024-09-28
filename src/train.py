import torch
import torch.optim as optim
from src.utils import sample_batch
from src.eval import evaluate_model


def train_model(model, dataset, train_data, val_data, val_labels, niter, ckpt_interval, batch_size,
                beta, lr, batch_inference, inference_batch_size, savepath, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    running_val_result = 0.0
    for iter in range(niter):
        model.train()
        sample = sample_batch(batch_size, train_data, device)
        optimizer.zero_grad()
        loss = model.compute_loss(sample, beta)
        loss.backward()
        optimizer.step()
        if (iter+1) % ckpt_interval == 0:
            val_result = evaluate_model(model, dataset, val_data, val_labels, beta, device,
                                        batch_inference, inference_batch_size)
            if val_result > running_val_result:
                torch.save(model.state_dict(), savepath)
                running_val_result = val_result
