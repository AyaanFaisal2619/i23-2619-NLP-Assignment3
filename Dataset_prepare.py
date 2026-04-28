import pandas as pd
import os
import gc

DATA_DIR_NAME = "Dataset"

rows_per_file = 10000  # only take 10k rows from each file


def resolve_data_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.getcwd(), DATA_DIR_NAME),
        os.path.join(script_dir, DATA_DIR_NAME),
        os.path.join(os.path.dirname(script_dir), DATA_DIR_NAME),
    ]
    fallback_existing_dir = None
    for candidate in candidates:
        if os.path.isdir(candidate):
            if fallback_existing_dir is None:
                fallback_existing_dir = candidate
            entries = os.listdir(candidate)
            if any(name.endswith(".json") for name in entries):
                return candidate
    if fallback_existing_dir is not None:
        return fallback_existing_dir
    raise FileNotFoundError(
        "Could not find Dataset directory. Checked current directory, script directory, and parent directory."
    )


def main():
    data_dir = resolve_data_dir()
    chunks = []

    files = os.listdir(data_dir)
    json_files = [f for f in files if f.endswith(".json")]
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {data_dir}")

    for file in json_files:
        path = os.path.join(data_dir, file)
        print(f"Reading {file}...")

        collected = 0

        for chunk in pd.read_json(path, lines=True, chunksize=5000):
            remaining = rows_per_file - collected

            if remaining <= 0:
                break

            chunk = chunk.head(remaining)

            chunk.drop(
                columns=[
                    "reviewerID",
                    "reviewerName",
                    "asin",
                    "helpful",
                    "unixReviewTime",
                    "reviewTime",
                    "summary",
                ],
                errors="ignore",
                inplace=True,
            )

            chunks.append(chunk)
            collected += len(chunk)

            print(f"Collected {collected}/{rows_per_file} rows from {file}")

        gc.collect()

    # Merge dataset
    final_df = pd.concat(chunks, ignore_index=True)
    print("Total dataset shape:", final_df.shape)

    # Shuffle before splitting
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    total = len(final_df)
    train_end = int(0.70 * total)
    val_end = int(0.85 * total)  # 70% + 15%

    train_df = final_df[:train_end]
    val_df = final_df[train_end:val_end]
    test_df = final_df[val_end:]

    print("Train:", train_df.shape)
    print("Val:", val_df.shape)
    print("Test:", test_df.shape)

    # Save files strictly inside Dataset/
    out_dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_DIR_NAME)
    os.makedirs(out_dataset_dir, exist_ok=True)
    train_path = os.path.join(out_dataset_dir, "dataset_train.csv")
    val_path = os.path.join(out_dataset_dir, "dataset_val.csv")
    test_path = os.path.join(out_dataset_dir, "dataset_test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Saved splits to Dataset/")


if __name__ == "__main__":
    main()