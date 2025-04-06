import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchinfo import summary
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """Attention mechanism to focus on relevant parts of the STI."""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        # Reduce channels significantly for attention map calculation
        inter_channels = max(1, in_channels // 8) # Ensure at least 1 channel
        self.conv_query = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1)) # Learnable scaling factor

    def forward(self, x):
        batch_size, C, H, W = x.size()

        proj_query = self.conv_query(x).view(batch_size, -1, W * H).permute(0, 2, 1)
        proj_key = self.conv_key(x).view(batch_size, -1, W * H)
        energy = torch.bmm(proj_query, proj_key) # Batch matrix multiplication
        attention = F.softmax(energy, dim=-1) # Softmax over keys

        proj_value = self.conv_value(x).view(batch_size, -1, W * H)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)

        out = self.gamma * out + x # Add residual connection scaled by gamma
        return out


class MLPHead(nn.Module):
    """Improved regression head using an MLP instead of a single linear layer."""
    def __init__(self, in_features, hidden_dim=128, dropout_rate=0.5):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim) # Use BatchNorm1d for features
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        # Ensure input to BatchNorm1d has more than 1 element if batch size > 1
        # If batch size is 1, BatchNorm might cause issues during training/eval
        if x.size(0) > 1:
             x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class Enhanced_HR_CNN(nn.Module):
    """
    Enhanced CNN model based on ResNet18 for Heart Rate estimation from STIs.
    Accepts 3-channel input (stack of STIs from different ROIs).
    Includes attention mechanisms, better feature extraction, and improved regression head.
    """
    def __init__(self, dropout_rate=0.5, use_pretrained=True):
        super(Enhanced_HR_CNN, self).__init__()

        # Load base model - now configurable to use pretrained or not
        weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
        resnet = models.resnet18(weights=weights)

        # --- Input Layer Modification --- 
        original_conv1 = resnet.conv1
        # Change input channels from 1 to 3
        self.conv1 = nn.Conv2d(3, original_conv1.out_channels,
                              kernel_size=original_conv1.kernel_size,
                              stride=original_conv1.stride,
                              padding=original_conv1.padding,
                              bias=(original_conv1.bias is not None))

        # Transfer weights properly if using pretrained model
        if use_pretrained and hasattr(resnet.conv1, 'weight'): # Check attribute exists
            # If using pretrained, directly copy weights (3 channels -> 3 channels)
            self.conv1.weight.data = resnet.conv1.weight.data
            if resnet.conv1.bias is not None:
                self.conv1.bias.data = resnet.conv1.bias.data

        # --- ResNet Backbone --- 
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2 # Output channels: 128

        # Add attention after mid-level features
        self.attention1 = AttentionBlock(128) # Input channels for AttentionBlock must match layer2 output

        self.layer3 = resnet.layer3 # Output channels: 256
        self.layer4 = resnet.layer4 # Output channels: 512

        # Add attention after high-level features
        self.attention2 = AttentionBlock(512) # Input channels for AttentionBlock must match layer4 output

        self.avgpool = resnet.avgpool

        # --- Multi-scale Features --- 
        # Use features from different levels of the network AFTER attention (if applied)
        # Reduce channels of deeper layers to a common dimension (e.g., 128)
        self.layer2_pool = nn.AdaptiveAvgPool2d(1)
        self.layer3_pool = nn.AdaptiveAvgPool2d(1)
        self.layer4_pool = nn.AdaptiveAvgPool2d(1)

        # --- Improved Regression Head --- 
        num_ftrs_layer4 = resnet.fc.in_features # 512 features from the main path (after layer4 attention & avgpool)
        num_ftrs_layer2 = 128 # Features from layer 2 path (after attention & avgpool)
        num_ftrs_layer3 = 256 # Features from layer 3 path (after avgpool)

        total_in_features = num_ftrs_layer4 + num_ftrs_layer2 + num_ftrs_layer3
        self.mlp_head = MLPHead(total_in_features, hidden_dim=256, dropout_rate=dropout_rate)

        # Initialize weights for new layers (Attention, MLPHead)
        # Note: ResNet layers retain their original initialization (or pretrained weights)
        # self._initialize_weights([self.attention1, self.attention2, self.mlp_head]) # Call helper

    # def _initialize_weights(self, modules_list):
    #     for modules in modules_list:
    #         for m in modules.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #                 if m.bias is not None:
    #                     nn.init.constant_(m.bias, 0)
    #             elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
    #                 nn.init.constant_(m.weight, 1)
    #                 nn.init.constant_(m.bias, 0)
    #             elif isinstance(m, nn.Linear):
    #                 nn.init.xavier_normal_(m.weight)
    #                 if m.bias is not None:
    #                     nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass with multi-scale feature extraction and attention.
        Args:
            x (torch.Tensor): Input tensor (batch of STIs).
                              Expected shape: (batch_size, 3, H_sti, W_sti)
        Returns:
            torch.Tensor: Output tensor (batch of predicted HRs). Shape: (batch_size, 1)
        """
        # Input normalization (standardize per-image)
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        std = torch.std(x, dim=(2, 3), keepdim=True)
        x = (x - mean) / (std + 1e-5)

        # Initial convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet backbone with feature extraction at multiple levels
        x1 = self.layer1(x)
        x2 = self.layer2(x1) # (N, 128, H/8, W/8)
        x2_att = self.attention1(x2)

        x3 = self.layer3(x2_att) # (N, 256, H/16, W/16)
        x4 = self.layer4(x3) # (N, 512, H/32, W/32)
        x4_att = self.attention2(x4)

        # Extract multi-scale features using adaptive pooling
        f2 = self.layer2_pool(x2_att).view(x.size(0), -1) # (N, 128)
        f3 = self.layer3_pool(x3).view(x.size(0), -1) # (N, 256)
        f4_main = self.avgpool(x4_att).view(x.size(0), -1) # (N, 512) - This is the main feature path

        # Concatenate all features
        # Features: Main path (layer4 attentuated), Layer3 path, Layer2 attentuated path
        combined_features = torch.cat([f4_main, f3, f2], dim=1) # (N, 512 + 256 + 128)

        # Regression head
        hr_output = self.mlp_head(combined_features)

        return hr_output


class HREstimationLoss(nn.Module):
    """Custom loss function for HR estimation combining L1, MSE, and correlation loss."""
    def __init__(self, mse_weight=1.0, l1_weight=0.5, corr_weight=0.3):
        super(HREstimationLoss, self).__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.corr_weight = corr_weight

    def forward(self, pred, target):
        # Ensure target is same shape as prediction
        if target.ndim == 1:
            target = target.unsqueeze(1)
            
        # MSE Loss
        mse_loss = F.mse_loss(pred, target)

        # L1 Loss (MAE)
        l1_loss = F.l1_loss(pred, target)

        # Correlation Loss (higher correlation is better)
        # Ensure non-zero variance before calculating correlation
        vx = pred - torch.mean(pred)
        vy = target - torch.mean(target)
        std_vx = torch.std(pred)
        std_vy = torch.std(target)
        
        if std_vx > 1e-5 and std_vy > 1e-5:
            corr_loss = -torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-6)
        else:
            # If variance is near zero, correlation is undefined or unstable; assign zero loss
            corr_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Combined loss
        total_loss = self.mse_weight * mse_loss + self.l1_weight * l1_loss + self.corr_weight * corr_loss

        return total_loss


class HR_Trainer:
    """Training class with advanced training features."""
    def __init__(self, model, optimizer=None, criterion=None, device=None, lr=1e-3, weight_decay=1e-4):
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = optimizer if optimizer is not None else optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = criterion if criterion is not None else HREstimationLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            # Ensure inputs are float32 and targets are appropriate type/shape
            inputs = inputs.to(self.device, dtype=torch.float32)
            targets = targets.to(self.device, dtype=torch.float32).unsqueeze(1) # Ensure target is [N, 1]

            # Apply data augmentation in training
            inputs = self._apply_augmentation(inputs)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Training Loss: {epoch_loss:.4f}') # Add print statement
        return epoch_loss

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device, dtype=torch.float32)
                targets = targets.to(self.device, dtype=torch.float32).unsqueeze(1) # Ensure target is [N, 1]

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)

                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

        # Calculate validation metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        val_loss = running_loss / len(val_loader.dataset)
        mae = F.l1_loss(all_preds, all_targets).item()
        mse = F.mse_loss(all_preds, all_targets).item()
        rmse = torch.sqrt(torch.tensor(mse)).item()

        print(f'Validation Loss: {val_loss:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}') # Add print statement

        # Update learning rate scheduler
        self.scheduler.step(val_loss)

        return val_loss, mae, rmse

    def _apply_augmentation(self, x):
        """Apply data augmentation for training."""
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, [3]) # Flip along width dimension

        # Random brightness adjustment (additive)
        # Apply carefully after standardization
        # if torch.rand(1).item() > 0.5:
        #     brightness_shift = (torch.rand(1).item() - 0.5) * 0.4 # Shift between -0.2 and 0.2
        #     x = x + brightness_shift

        # Random noise addition
        if torch.rand(1).item() > 0.5:
            noise = torch.randn_like(x) * 0.05 # Small noise relative to std=1
            x = x + noise

        # Clip values if necessary after augmentations
        # x = torch.clamp(x, -3.0, 3.0) # Example clipping if needed

        return x


# Example usage (for testing the enhanced model)
if __name__ == '__main__':
    # Create a dummy input tensor representing a batch of stacked STIs
    # Shape: (batch_size, channels, height, width) = (4, 3, 224, 224)
    dummy_sti = torch.randn(4, 3, 224, 224) # Use batch size > 1 and 3 channels

    # Instantiate the enhanced model
    model = Enhanced_HR_CNN(dropout_rate=0.5, use_pretrained=False) # Test without pretrained

    # Print model summary
    try:
        # Note: torchinfo might require batch_size in input_size tuple
        summary(model, input_size=(4, 3, 224, 224)) # Update input_size
    except ImportError:
        print("torchinfo not found, skipping model summary.")
        print(model)

    # Test forward pass
    model.train() # Set to train mode to test BatchNorm/Dropout behavior
    predicted_hr = model(dummy_sti)

    print(f"\nInput shape: {dummy_sti.shape}")
    print(f"Output shape: {predicted_hr.shape}")
    print(f"Predicted HR (dummy input, train mode):\n{predicted_hr}")

    # Test loss function
    dummy_target = torch.randn(4, 1) * 10 + 70 # Example HR targets
    criterion = HREstimationLoss()
    loss = criterion(predicted_hr, dummy_target)
    print(f"\nCalculated Loss (dummy data): {loss.item():.4f}")

    # Initialize trainer and test methods (optional)
    # Note: Requires dummy DataLoader setup for train_epoch/validate
    # optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    # trainer = HR_Trainer(model, optimizer, criterion)
    # print(f"\nTrainer initialized on device: {trainer.device}")

    # Print model parameters
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
