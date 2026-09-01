import torch

from model import create_model


model = create_model(
    num_classes=5
)

print(
    "Model created successfully."
)

print(
    "\nClassifier:"
)

print(
    model.classifier
)


# Test a fake batch

x = torch.randn(
    2,
    3,
    224,
    224
)

with torch.no_grad():

    output = model(x)


print(
    "\nInput shape:",
    x.shape
)

print(
    "Output shape:",
    output.shape
)

print(
    "Output:",
    output
)