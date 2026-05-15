import torch

from mnar_net import MDCTBackbone


def main() -> None:
    model = MDCTBackbone(num_classes=9, signal_length=1024)
    inputs = torch.randn(4, 3, 1024)
    logits = model(inputs)
    features = model.get_features(inputs)

    print("logits:", tuple(logits.shape))
    print("features:", tuple(features.shape))


if __name__ == "__main__":
    main()
