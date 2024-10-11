import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from praxis.PointingTransformer import YOLOBackbone, PointingDeviceClassification

# Define number of classes for your dataset
num_classes = 3  # Example: 3 object categories
transformer_hidden_dim = 256 #512
num_transformer_layers = 4 #6
learning_rate = 1e-4

# Instantiate the DeepPoint model and the combined model with YOLO backbone
#deep_point_model = DeepPoint()
pointing_classification_model = PointingDeviceClassification(num_classes, transformer_hidden_dim,
                                                                    num_transformer_layers)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move the model to the correct device
pointing_classification_model = pointing_classification_model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(pointing_classification_model.parameters(), lr=learning_rate)

# Example training data
train_loader = [(torch.rand(8, 3, 224, 224), torch.rand(8, 3), torch.randint(0, num_classes, (8,))) for _ in
                range(100)]  # Simulated data

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    pointing_classification_model.enable_training()
    running_loss = 0.0
    for i, (images, pointing_vectors, labels) in enumerate(train_loader):
        images = images.to(device)
        pointing_vectors = pointing_vectors.to(device)
        # Forward pass
        outputs = pointing_classification_model(images, pointing_vectors)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 mini-batches
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

print('Finished Training')
