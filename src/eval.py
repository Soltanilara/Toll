import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score


def evaluate_model(model, dataset, test_data, true_labels, beta, device, batch_inference, inference_batch_size):
    model.eval()
    with torch.no_grad():
        queries = torch.from_numpy(test_data).float().to(device)

        # Get model output in batches if memory becomes an issue
        if batch_inference:
            num_samples = queries.size(0)
            recon_list = []
            z_list = []
            for i in range(0, num_samples, inference_batch_size):
                batch_queries = queries[i:min(i + inference_batch_size, num_samples)]
                recon_batch, z_batch = model(batch_queries)
                recon_list.append(recon_batch)
                z_list.append(z_batch)
            # Concatenate all batch outputs
            recon = torch.cat(recon_list, dim=0)
            z = torch.cat(z_list, dim=0)
        else:
            recon, z = model(queries)

        recon_loss = torch.norm(recon.view(recon.size(0), -1) - queries.view(queries.size(0), -1), dim=1)
        z_norm = torch.norm(z, dim=1)
        scores = recon_loss + beta * z_norm

        if dataset != 'arrhythmia':
            res = roc_auc_score(true_labels, scores.cpu().numpy())
        else:
            preds = np.ones(scores.size(0),dtype=int)
            preds[np.argsort(scores.cpu().numpy())[:np.round(0.854 * scores.size(0)).astype(int)]] = 0
            res = f1_score(true_labels, preds, average='binary')
    return res
