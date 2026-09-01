import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


def create_model(num_classes=5):

    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(
        weights=weights
    )

    # -----------------------------------------------------
    # IMPORTANT
    # -----------------------------------------------------
    # Do NOT freeze the feature extractor here.
    #
    # Grad-CAM needs gradients through the convolutional
    # feature layers.
    #
    # During training we can freeze layers explicitly in
    # the training script if required.
    # -----------------------------------------------------

    input_features = (
        model.classifier[1].in_features
    )

    model.classifier[1] = nn.Linear(
        input_features,
        num_classes
    )

    return model