import torch

from mnar_net import MNARNet


def main() -> None:
    model = MNARNet(num_classes=3, signal_length=1024, memory_per_class=5)

    x_clean = torch.randn(12, 3, 1024)
    x_noisy = torch.randn(12, 3, 1024)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)

    model.update_memory(x_clean=x_clean, x_noisy=x_noisy, labels=labels)
    replay_x, replay_y = model.sample_replay(snr_db=-10.0, seed=42, use_adaptive_noise=True)

    model.expand_classifier(new_num_classes=5)

    print("replay_x:", tuple(replay_x.shape))
    print("replay_y:", tuple(replay_y.shape))
    print("memory_size:", len(model.memory))


if __name__ == "__main__":
    main()
