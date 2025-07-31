"""Neural network models for WORDLE solving - Phase 3 implementation.

This module provides sophisticated neural network architectures for         for _i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear) and layer_idx < len(self.residual_layers):
                # Apply residual connection
                residual = self.residual_layers[layer_idx](current if layer_idx == 0 else previous_x)
                current = layer(current)

                # Add residual if dimensions match
                if residual.shape == current.shape:
                    current = current + residual

                layer_idx += 1
                previous_x = current
            else:
                current = layer(current)imal WORDLE guesses using deep learning approaches.
"""

import logging
from typing import Any

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from .. import MLModelError
from .models import WordleMLModel

logger = logging.getLogger(__name__)


class WordleDataset(Dataset):
    """Dataset class for WORDLE neural network training."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Initialize dataset.

        Args:
            features: Feature matrix of shape (n_samples, n_features)
            targets: Target values of shape (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        return self.features[idx], self.targets[idx]


class DeepWordleNet(nn.Module):
    """Deep neural network for WORDLE guess prediction.

    Multi-layer feedforward network with dropout, batch normalization,
    and residual connections for robust word prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        use_residual: bool = True
    ) -> None:
        """Initialize the deep neural network.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual

        # Build network layers
        layers = []
        prev_dim = input_dim

        for _i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(nn.ReLU())

            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

        # Residual connections (if enabled and compatible dimensions)
        self.residual_layers = nn.ModuleList()
        if use_residual:
            for i, hidden_dim in enumerate(hidden_dims):
                if i == 0:
                    # First residual: input -> hidden
                    if input_dim == hidden_dim:
                        self.residual_layers.append(nn.Identity())
                    else:
                        self.residual_layers.append(nn.Linear(input_dim, hidden_dim))
                else:
                    # Subsequent residuals: hidden -> hidden
                    prev_hidden = hidden_dims[i-1]
                    if prev_hidden == hidden_dim:
                        self.residual_layers.append(nn.Identity())
                    else:
                        self.residual_layers.append(nn.Linear(prev_hidden, hidden_dim))

        logger.debug(f"DeepWordleNet initialized with {sum(p.numel() for p in self.parameters())} parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output predictions of shape (batch_size, 1)
        """
        if not self.use_residual:
            return self.network(x)

        # Forward pass with residual connections
        current = x
        layer_idx = 0
        previous_x = x

        for _i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear) and layer_idx < len(self.residual_layers):
                # Apply residual connection
                residual = self.residual_layers[layer_idx](current if layer_idx == 0 else previous_x)
                current = layer(current)

                # Add residual (if dimensions match)
                if current.shape == residual.shape:
                    current = current + residual

                layer_idx += 1
                previous_x = current
            else:
                current = layer(current)

        return current


class CNNWordleNet(nn.Module):
    """Convolutional neural network for WORDLE pattern recognition.

    Uses 1D convolutions to capture local patterns in word features
    and letter positions.
    """

    def __init__(
        self,
        input_dim: int,
        num_filters: list[int] | None = None,
        filter_sizes: list[int] | None = None,
        pool_sizes: list[int] | None = None,
        fc_dims: list[int] | None = None,
        dropout_rate: float = 0.3
    ) -> None:
        """Initialize CNN network.

        Args:
            input_dim: Number of input features
            num_filters: Number of filters for each conv layer
            filter_sizes: Filter sizes for each conv layer
            pool_sizes: Pool sizes for each conv layer
            fc_dims: Fully connected layer dimensions
            dropout_rate: Dropout probability
        """
        super().__init__()

        if num_filters is None:
            num_filters = [64, 32, 16]
        if filter_sizes is None:
            filter_sizes = [3, 3, 3]
        if pool_sizes is None:
            pool_sizes = [2, 2, 2]
        if fc_dims is None:
            fc_dims = [128, 64]

        self.input_dim = input_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.pool_sizes = pool_sizes
        self.fc_dims = fc_dims
        self.dropout_rate = dropout_rate

        # Conv layers
        conv_layers = []
        in_channels = 1  # Single channel input

        for _i, (out_channels, filter_size, pool_size) in enumerate(
            zip(num_filters, filter_sizes, pool_sizes, strict=False)
        ):
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, filter_size, padding=filter_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(pool_size),
                nn.Dropout(dropout_rate)
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate conv output size
        conv_output_size = input_dim
        for pool_size in pool_sizes:
            conv_output_size = conv_output_size // pool_size

        flatten_size = num_filters[-1] * conv_output_size

        # Fully connected layers
        fc_layers = []
        prev_dim = flatten_size

        for fc_dim in fc_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, fc_dim),
                nn.BatchNorm1d(fc_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = fc_dim

        # Output layer
        fc_layers.append(nn.Linear(prev_dim, 1))
        fc_layers.append(nn.Sigmoid())

        self.fc_layers = nn.Sequential(*fc_layers)

        logger.debug(f"CNNWordleNet initialized with {sum(p.numel() for p in self.parameters())} parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output predictions of shape (batch_size, 1)
        """
        # Reshape for 1D conv: (batch_size, 1, input_dim)
        x = x.unsqueeze(1)

        # Conv layers
        x = self.conv_layers(x)

        # Flatten for FC layers
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.fc_layers(x)

        return x


class NeuralWordleModel(WordleMLModel):
    """Neural network model for WORDLE solving.

    Supports multiple architectures (feedforward, CNN) with advanced
    training features like learning rate scheduling and early stopping.
    """

    def __init__(
        self,
        architecture: str = "deep",
        input_dim: int | None = None,
        **model_kwargs
    ) -> None:
        """Initialize neural model.

        Args:
            architecture: Model architecture ("deep" or "cnn")
            input_dim: Number of input features
            **model_kwargs: Additional model parameters
        """
        self.architecture = architecture
        self.input_dim = input_dim
        self.model_kwargs = model_kwargs

        self.model: nn.Module | None = None
        self.optimizer: optim.Optimizer | None = None
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._is_trained = False
        self._training_history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": []
        }

        logger.debug(f"NeuralWordleModel initialized with {architecture} architecture")

    def _build_model(self, input_dim: int) -> nn.Module:
        """Build the neural network model.

        Args:
            input_dim: Number of input features

        Returns:
            Neural network model

        Raises:
            MLModelError: If architecture is not supported
        """
        if self.architecture == "deep":
            return DeepWordleNet(input_dim, **self.model_kwargs)
        elif self.architecture == "cnn":
            return CNNWordleNet(input_dim, **self.model_kwargs)
        else:
            raise MLModelError(f"Unsupported architecture: {self.architecture}")

    def train(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        validation_split: float = 0.2,
        batch_size: int = 64,
        epochs: int = 100,
        learning_rate: float = 0.001,
        patience: int = 10,
        min_delta: float = 1e-4
    ) -> None:
        """Train the neural network model.

        Args:
            features: Feature matrix of shape (n_samples, n_features)
            targets: Target values of shape (n_samples,)
            validation_split: Fraction of data for validation
            batch_size: Training batch size
            epochs: Maximum number of epochs
            learning_rate: Initial learning rate
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping

        Raises:
            MLModelError: If training fails
        """
        try:
            logger.info(f"Starting neural network training with {features.shape[0]} samples")

            # Initialize model if not already done
            if self.model is None:
                self.input_dim = features.shape[1]
                self.model = self._build_model(self.input_dim)
                self.model.to(self.device)

                # Initialize optimizer
                self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

            # Split data
            n_samples = features.shape[0]
            n_val = int(n_samples * validation_split)
            indices = np.random.permutation(n_samples)

            train_indices = indices[n_val:]
            val_indices = indices[:n_val]

            train_features = features[train_indices]
            train_targets = targets[train_indices]
            val_features = features[val_indices]
            val_targets = targets[val_indices]

            # Create datasets and loaders
            train_dataset = WordleDataset(train_features, train_targets)
            val_dataset = WordleDataset(val_features, val_targets)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
            )

            # Training loop with early stopping
            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0

                for batch_features, batch_targets in train_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(batch_features).squeeze()
                    loss = self.criterion(outputs, batch_targets)
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # Validation phase
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_features, batch_targets in val_loader:
                        batch_features = batch_features.to(self.device)
                        batch_targets = batch_targets.to(self.device)

                        outputs = self.model(batch_features).squeeze()
                        loss = self.criterion(outputs, batch_targets)
                        val_loss += loss.item()

                val_loss /= len(val_loader)

                # Update learning rate
                scheduler.step(val_loss)

                # Record history
                self._training_history["train_loss"].append(train_loss)
                self._training_history["val_loss"].append(val_loss)
                self._training_history["learning_rate"].append(
                    self.optimizer.param_groups[0]['lr']
                )

                # Early stopping check
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs}: "
                        f"Train Loss: {train_loss:.6f}, "
                        f"Val Loss: {val_loss:.6f}, "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                    )

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            self._is_trained = True
            logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")

        except Exception as e:
            logger.error(f"Neural network training failed: {e}")
            raise MLModelError(f"Neural network training failed: {e}") from e

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions using the neural network.

        Args:
            features: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples,)

        Raises:
            MLModelError: If prediction fails or model not trained
        """
        if not self._is_trained or self.model is None:
            raise MLModelError("Model must be trained before making predictions")

        try:
            self.model.eval()
            features_tensor = torch.FloatTensor(features).to(self.device)

            with torch.no_grad():
                predictions = self.model(features_tensor).squeeze()
                predictions = predictions.cpu().numpy()

            # Ensure correct shape
            if predictions.ndim == 0:
                predictions = np.array([predictions])

            return predictions

        except Exception as e:
            logger.error(f"Neural network prediction failed: {e}")
            raise MLModelError(f"Neural network prediction failed: {e}") from e

    def save_model(self, filepath: str) -> None:
        """Save neural network model to file.

        Args:
            filepath: Path to save model

        Raises:
            MLModelError: If saving fails
        """
        if not self._is_trained or self.model is None:
            raise MLModelError("Cannot save untrained model")

        try:
            model_data = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'architecture': self.architecture,
                'input_dim': self.input_dim,
                'model_kwargs': self.model_kwargs,
                'training_history': self._training_history,
                'is_trained': self._is_trained,
                'model_type': 'NeuralWordleModel'
            }

            torch.save(model_data, filepath)
            logger.info(f"Neural model saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save neural model: {e}")
            raise MLModelError(f"Failed to save neural model: {e}") from e

    def load_model(self, filepath: str) -> None:
        """Load neural network model from file.

        Args:
            filepath: Path to load model from

        Raises:
            MLModelError: If loading fails
        """
        try:
            model_data = torch.load(filepath, map_location=self.device)

            if model_data['model_type'] != 'NeuralWordleModel':
                raise MLModelError("Model type mismatch: expected NeuralWordleModel")

            # Restore model parameters
            self.architecture = model_data['architecture']
            self.input_dim = model_data['input_dim']
            self.model_kwargs = model_data['model_kwargs']
            self._training_history = model_data.get('training_history', {})
            self._is_trained = model_data['is_trained']

            # Rebuild and load model
            self.model = self._build_model(self.input_dim)
            self.model.load_state_dict(model_data['model_state_dict'])
            self.model.to(self.device)

            # Restore optimizer if available
            if model_data.get('optimizer_state_dict'):
                self.optimizer = optim.Adam(self.model.parameters())
                self.optimizer.load_state_dict(model_data['optimizer_state_dict'])

            logger.info(f"Neural model loaded from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load neural model: {e}")
            raise MLModelError(f"Failed to load neural model: {e}") from e

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the neural model.

        Returns:
            Dictionary with model information
        """
        info = {
            'model_type': 'NeuralWordleModel',
            'architecture': self.architecture,
            'is_trained': self._is_trained,
            'input_dim': self.input_dim,
            'device': str(self.device),
            'model_kwargs': self.model_kwargs
        }

        if self.model is not None:
            info['num_parameters'] = sum(p.numel() for p in self.model.parameters())
            info['trainable_parameters'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        if self._training_history:
            info['training_epochs'] = len(self._training_history.get('train_loss', []))
            if self._training_history.get('val_loss'):
                info['best_val_loss'] = min(self._training_history['val_loss'])

        return info

    def get_training_history(self) -> dict[str, list[float]]:
        """Get training history for analysis.

        Returns:
            Dictionary with training metrics over epochs
        """
        return self._training_history.copy()
