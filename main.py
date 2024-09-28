import os
import argparse
import numpy as np
import torch
from src.data import load_mnist, load_cifar, load_arrhythmia
from src.models import build_mnist_model, build_cifar_model, build_arrhythmia_model
from src.train import train_model
from src.eval import evaluate_model

def main():

    # Arguments
    parser = argparse.ArgumentParser(description='Train and evaluate Toll')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist','fmnist','cifar10','cifar100','arrhythmia'])
    parser.add_argument('--dataset_path', type=str, default='mnist', help='path to dataset files')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--num_seeds', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--niter', type=int, default=2000)
    parser.add_argument('--ckpt_interval', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--z_dim', type=int, default=128, help='bottleneck dimension')
    parser.add_argument('--beta', type=float, default=1000)
    parser.add_argument('--batch_inference', type=bool, default=False,
                        help='Whether during inference samples are passed through the network in batches or all at once')
    parser.add_argument('--inference_batch_size', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_path', type=str, default='output')
    args = parser.parse_args()

    experiment_name = f"{args.dataset}_lr{args.lr}_batch{args.batch_size}_beta{args.beta}_z_dim{args.z_dim}" \
                      f"_niter{args.niter}_seeds{args.num_seeds}"
    experiment_dir = os.path.join(args.output_path,experiment_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    np.random.seed(42)

    # Initialize an array for results
    results = np.zeros((args.num_seeds,args.num_classes))

    # Load data
    if args.dataset in {'mnist', 'fmnist'}:
        train_data, train_labels, val_data, val_labels, test_data, test_labels = \
            load_mnist(args.dataset, args.dataset_path)
    elif args.dataset in {'cifar10', 'cifar100'}:
        train_data, train_labels, val_data, val_labels, test_data, test_labels, min_max = \
            load_cifar(args.dataset, args.dataset_path)
    elif args.dataset == 'arrhythmia':
        X_train, X_val, X_test, y_val, y_test = \
            load_arrhythmia(args.dataset_path)

    # Loop aver random network initializations
    for torchseed in range(args.num_seeds):
        torch.manual_seed(torchseed)
        np.random.seed(42)

        # Loop over classes
        for normalclass in range(args.num_classes):
            # Build model
            if args.dataset in {'mnist', 'fmnist'}:
                model = build_mnist_model(args.z_dim, args.dataset).to(args.device)
            elif args.dataset in {'cifar10', 'cifar100'}:
                model = build_cifar_model(args.z_dim, args.dataset).to(args.device)
            elif args.dataset == 'arrhythmia':
                model = build_arrhythmia_model(args.z_dim, args.dataset).to(args.device)

            ckpt = os.path.join(experiment_dir, f'model_seed{torchseed}_class{normalclass}.pth')  # Path to saved models

            if args.dataset != 'arrhythmia':
                X_test,y_test = test_data.copy(), np.ones_like(test_labels, dtype=int)
                y_test[test_labels == normalclass] = 0

                if args.dataset in {'cifar10', 'cifar100'}:
                    # Min-max scaling
                    X_test -= np.array([min_max[normalclass][0]] * 3).reshape(1, 3, 1, 1)
                    X_test /= np.array([min_max[normalclass][1] - min_max[normalclass][0]] * 3).reshape(1, 3, 1, 1)

            if args.mode == 'train':
                if args.dataset != 'arrhythmia':
                    X_train, X_val = train_data[train_labels == normalclass], val_data.copy()
                    y_val = np.ones_like(val_labels, dtype=int)
                    y_val[val_labels == normalclass] = 0

                    if args.dataset in {'cifar10', 'cifar100'}:
                        # Min-max scaling
                        X_train -= np.array([min_max[normalclass][0]] * 3).reshape(1, 3, 1, 1)
                        X_train /= np.array([min_max[normalclass][1] - min_max[normalclass][0]] * 3).reshape(1, 3, 1, 1)
                        X_val -= np.array([min_max[normalclass][0]] * 3).reshape(1, 3, 1, 1)
                        X_val /= np.array([min_max[normalclass][1] - min_max[normalclass][0]] * 3).reshape(1, 3, 1, 1)

                # Training
                train_model(model, args.dataset, X_train, X_val, y_val, args.niter, args.ckpt_interval,
                            args.batch_size, args.beta, args.lr, args.batch_inference,
                            args.inference_batch_size, ckpt, args.device)

            # Evaluation
            model.load_state_dict(torch.load(ckpt, weights_only=True))
            # model.load_state_dict(torch.load(ckpt))
            results[torchseed, normalclass] = evaluate_model(model, args.dataset, X_test, y_test, args.beta, args.device,
                                                             args.batch_inference, args.inference_batch_size)
            if args.dataset != 'arrhythmia':
                print(f'Seed: {torchseed+1}/{args.num_seeds} | Normal Class: {normalclass} | '
                      f'Test AUROC: {results[torchseed, normalclass]*100:.1f}')
            else:
                print(f'Seed: {torchseed + 1}/{args.num_seeds} | '
                      f'Test F1: {results[torchseed, normalclass] * 100:.1f}')

    # Calculate averaged results and standard deviations over seeds
    averaged_results = np.mean(results,axis=0)*100
    standard_deviations = np.std(results,axis=0)*100

    # Print out the results
    if args.dataset != 'arrhythmia':
        print(f'Mean test AUROC and standard deviations over {args.num_seeds} seeds (%):')
        for class_id in range(args.num_classes):
            print(f'Class {class_id}:    Mean: {averaged_results[class_id]:.1f} | '
                  f'SD: {standard_deviations[class_id]:.1f}')
        print(f'Dataset average:  {np.mean(averaged_results):.1f}')
    else:
        print(f'Mean test F1 and standard deviations over {args.num_seeds} seeds (%):')
        print(f'Mean F1: {averaged_results[0]:.1f} | SD: {standard_deviations[0]:.1f}')

if __name__ == '__main__':
    main()
