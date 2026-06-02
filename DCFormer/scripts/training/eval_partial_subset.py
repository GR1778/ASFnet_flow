import argparse
import importlib
import pickle
from pathlib import Path

import numpy as np

import train as train_module

human36m_dataset = importlib.import_module("mvn.datasets.human36m")


def load_labels(labels_path):
    with open(labels_path, "rb") as f:
        return pickle.load(f)


def build_subset_metadata(labels):
    subjects = sorted({int(label["subject"]) for label in labels})
    action_trials = sorted({
        ((int(label["action"]) - 2) * 2 + (int(label["subaction"]) - 1), int(label["action"]), int(label["subaction"]))
        for label in labels
    })

    original_action_names = list(human36m_dataset.retval["action_names"])
    subject_names = [f"S{subject}" for subject in subjects]
    action_names = [original_action_names[old_idx] for old_idx, _, _ in action_trials]
    action_idx_map = {old_idx: new_idx for new_idx, (old_idx, _, _) in enumerate(action_trials)}

    return subject_names, action_names, action_idx_map


def patch_human36m_for_subset(subject_names, action_names, action_idx_map):
    human36m_dataset.retval["subject_names"] = subject_names
    human36m_dataset.retval["action_names"] = action_names

    original_init = human36m_dataset.Human36MSingleViewDataset.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.labels_action_idx = np.array([action_idx_map[int(idx)] for idx in self.labels_action_idx], dtype=np.int64)

    human36m_dataset.Human36MSingleViewDataset.__init__ = patched_init


def patch_train_setup_dataloaders(args):
    original_setup = train_module.setup_dataloaders

    def patched_setup_dataloaders(config, is_train=True, distributed_train=False, rank=None, world_size=None):
        if args.eval and args.eval_dataset == "val":
            is_train = False
        return original_setup(config, is_train=is_train, distributed_train=distributed_train, rank=rank, world_size=world_size)

    train_module.setup_dataloaders = patched_setup_dataloaders


def parse_wrapper_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--subset-labels",
        type=str,
        default=None,
        help="Optional override for the subset labels path. Defaults to config.dataset.val_labels_path in eval mode.",
    )
    args, remaining = parser.parse_known_args()
    return args, remaining


def main():
    wrapper_args, remaining_argv = parse_wrapper_args()

    # Delegate the regular CLI to train.py after stripping wrapper-only args.
    import sys

    sys.argv = [sys.argv[0], *remaining_argv]
    args = train_module.parse_args()
    train_module.args = args

    if not args.eval:
        raise ValueError("eval_partial_subset.py is intended for --eval runs only.")

    labels_path = wrapper_args.subset_labels
    if labels_path is None:
        labels_path = train_module.config.dataset.val_labels_path if args.eval_dataset == "val" else train_module.config.dataset.train_labels_path

    labels = load_labels(Path(labels_path))
    subject_names, action_names, action_idx_map = build_subset_metadata(labels)

    print(f"Partial eval subjects: {subject_names}")
    print(f"Partial eval action trials: {len(action_names)}")

    patch_human36m_for_subset(subject_names, action_names, action_idx_map)
    patch_train_setup_dataloaders(args)

    train_module.main(args)


if __name__ == "__main__":
    main()
