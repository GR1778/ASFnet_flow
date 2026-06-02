import argparse
import pickle
import random


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.loads(f.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/h36m_train.pkl")
    parser.add_argument("--val", default="data/h36m_validation.pkl")
    parser.add_argument("--out", default="data/h36m_train_plus_val.pkl")
    parser.add_argument("--val-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = load_pickle(args.train)
    val = load_pickle(args.val)
    if not 0.0 <= args.val_fraction <= 1.0:
        raise ValueError("--val-fraction must be in [0, 1]")

    rng = random.Random(args.seed)
    n_val = int(round(len(val) * args.val_fraction))
    val_subset = rng.sample(val, n_val) if n_val < len(val) else list(val)
    merged = list(train) + val_subset

    with open(args.out, "wb") as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("train:", len(train))
    print("val:", len(val))
    print("val_fraction:", args.val_fraction)
    print("val_used:", len(val_subset))
    print("merged:", len(merged))
    print("saved:", args.out)


if __name__ == "__main__":
    main()
