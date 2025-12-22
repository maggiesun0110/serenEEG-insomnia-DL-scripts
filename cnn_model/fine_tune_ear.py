checkpoint = torch.load("checkpoints/sereneeg_cnn_v1.pth", map_location=device)

model = BaseEEGCNN().to(device)
model.load_state_dict(checkpoint["model_state_dict"])

# Freeze convolutional layers
for param in model.conv1.parameters():
    param.requires_grad = False
for param in model.conv2.parameters():
    param.requires_grad = False
for param in model.conv3.parameters():
    param.requires_grad = False

# Keep classifier trainable
for param in model.fc1.parameters():
    param.requires_grad = True
for param in model.fc2.parameters():
    param.requires_grad = True