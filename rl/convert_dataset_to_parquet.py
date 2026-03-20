"""
Convert DAPO-Math-17k dataset to parquet format for verl training
"""

import os
import pandas as pd
from datasets import load_dataset
import argparse


def convert_dapo_math_to_parquet(output_dir: str = "/root/SeedMitigating/data"):
    """
    Convert DAPO-Math-17k dataset to parquet format required by verl.

    verl expects parquet files with the following fields:
    - prompt: The input prompt/question
    - reward_model.ground_truth: The ground truth answer (for reward computation)
    - data_source: Source identifier (optional)
    """
    print("=" * 80)
    print("Converting DAPO-Math-17k to parquet format for verl")
    print("=" * 80)

    # Set HF mirror
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset with mirror
    print("\nLoading DAPO-Math-17k dataset...")
    dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", split="train")
    print(f"Loaded {len(dataset)} samples")
    print(f"Fields: {dataset.column_names}")

    # Split into train/val (90/10 split)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    print(f"\nSplitting dataset:")
    print(f"  Train: {train_size} samples")
    print(f"  Val: {val_size} samples")

    # Convert to pandas DataFrame
    df = pd.DataFrame(dataset)

    # Rename columns to match verl's expected format
    # DAPO-Math has: prompt, solution
    # verl expects: prompt, reward_model.ground_truth
    df_verl = pd.DataFrame({
        "prompt": df["prompt"],
        "reward_model.ground_truth": df["solution"],
        "data_source": ["math_dapo"] * len(df),
    })

    # Split
    train_df = df_verl.iloc[:train_size]
    val_df = df_verl.iloc[train_size:]

    # Save to parquet
    train_path = os.path.join(output_dir, "dapo_math_train.parquet")
    val_path = os.path.join(output_dir, "dapo_math_val.parquet")

    print(f"\nSaving parquet files...")
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")

    # Verify
    print(f"\nVerifying parquet files...")
    train_verify = pd.read_parquet(train_path)
    val_verify = pd.read_parquet(val_path)

    print(f"  Train shape: {train_verify.shape}")
    print(f"  Val shape: {val_verify.shape}")
    print(f"  Train columns: {list(train_verify.columns)}")

    print("\n" + "=" * 80)
    print("✓ Dataset conversion complete!")
    print("=" * 80)

    return train_path, val_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/root/SeedMitigating/data",
                       help="Output directory for parquet files")
    args = parser.parse_args()

    convert_dapo_math_to_parquet(args.output_dir)
