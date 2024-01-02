import torch
import torch.nn as nn

def create_model(input_length):
    # Define the model architecture as a list of layers
    layers = [
        nn.Conv1d(in_channels=1, out_channels=128, kernel_size=50, stride=6, padding=1, bias=False),  # Adjusted stride, added padding
        nn.BatchNorm1d(num_features=128),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=8, stride=8),
        nn.Dropout(p=0.5),
        nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=1, bias=False),  # Added padding
        nn.BatchNorm1d(num_features=128),
        nn.ReLU(inplace=True),
        nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=1, bias=False),  # Added padding
        nn.BatchNorm1d(num_features=128),
        nn.ReLU(inplace=True),
        nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, padding=1, bias=False),  # Added padding
        nn.BatchNorm1d(num_features=128),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=4, stride=4),
        nn.Dropout(p=0.5),
        nn.Flatten(),
    ]
    
    # Create a temporary model to calculate in_features
    temp_model = nn.Sequential(*layers)
    
    # Pass a dummy input through the model to calculate the output size
    dummy_input = torch.randn(1, 1, input_length)
    print(f"Input shape: {dummy_input.shape}")
    with torch.no_grad():
        for layer in temp_model:
            dummy_input = layer(dummy_input)
            print(f"Output shape after layer {type(layer).__name__}: {dummy_input.shape}")
            if dummy_input.nelement() == 0:
                raise ValueError(f"The output size is too small after {type(layer).__name__}.")

    # Calculate in_features from the output of the last layer before the Linear layer
    in_features = dummy_input.shape[1]
    print(in_features)
    # Add the final Linear layer with the correct in_features
    layers.append(nn.Linear(in_features, out_features=5, bias=False))
    
    # Create the full model with the Linear layer
    model = nn.Sequential(*layers)
    return model

# Define the input length and create the model
input_length = 3000
model = create_model(input_length)

# Verify that the model can process an input of the defined length
fake_x = torch.randn(2, 1, input_length)  # Example input of batch size 2
y_pred = model(fake_x)

# Print the shapes to verify the input and output
print(f"Input shape: {fake_x.shape}")
print(f"Output shape: {y_pred.shape}")
