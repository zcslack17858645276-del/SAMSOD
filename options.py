import argparse

def get_argparser():
    parser = argparse.ArgumentParser(description="SAM2 Finetuning Script")
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training and inference (e.g., 'cuda' or 'cpu').",
    )

    parser.add_argument(
        "--augment",
        type=bool,
        default=True,
        help="Turn on the data augment(RandomRotate and RandomFlip)"
    )

    parser.add_argument(
        "--model_cfg",
        type=str,
        default="configs/sam2/sam2_hiera_t.yaml",
        help="Path to the SAM2 model config file.",
    )
    
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="checkpoints/sam2_hiera_tiny.pt",
        help="Path to the SAM2 model checkpoint.",
    )
    
    parser.add_argument(
        "--train_data_root",
        type=str,
        default="dataset/DUTS-TR",
        help="Root directory of the training dataset.",
    )

    parser.add_argument(
        "--val_data_root",
        type=str,
        nargs="+",
        default=["dataset/DUTS-TE", "dataset/ECSSD", "dataset/HKU-IS", "dataset/PASCAL-S"],
        help="Root directory of the validation dataset.",
    )
    
    parser.add_argument(
        "--batch_size_train",
        type=int,
        default=2,
        help="Batch size for training.",
    )

    parser.add_argument(
        "--batch_size_val",
        type=int,
        default=1,
        help="Batch size for evaluation.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for data loading.",
    )

    parser.add_argument(
        "--input_size",
        type=int,
        default=1024,
        help="Input size for the SAM2 model.",
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer.",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay for the optimizer.",
    )
    
    parser.add_argument(
        "--print_interval",
        type=int,
        default=100,
        help="Number of iterations between printing training status.",
    )

    parser.add_argument(
        "--save_interval",
        type=int,
        default=1000,
        help="Number of iterations between saving model checkpoints.",
    )
    
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints_finetuned",
        help="Directory to save model checkpoints.",
    )

    parser.add_argument(
        "--predict_checkpoint",
        type=str,
        default="checkpoints_finetuned/sam2_final.pt",
        help="Path to the checkpoint for prediction.",
    )

    parser.add_argument(
        "--bce_weight",
        type=float,
        default=1.0,
        help="Weight for the bce loss"
    )

    parser.add_argument(
        "--dice_weight",
        type=float,
        default=1.0,
        help="Weight for the dice loss"
    )

    parser.add_argument(
        "--ssim_weight",
        type=float,
        default=1.0,
        help="Weight for the ssim loss"
    )


    
    return parser.parse_args()