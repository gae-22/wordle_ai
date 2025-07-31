"""Neural network models for WORDLE solving - Phase 3 implementation.

This module provides sophisticated neural network architectures for
optimal WORDLE guesses using deep learning approaches.
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

        for i, hidden_dim in enumerate(hidden_dims):
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

        self.network = nn.ModuleList(layers)

        # Residual connections (if enabled)
        self.residual_layers = nn.ModuleList()
        if use_residual:
            for hidden_dim in hidden_dims:
                self.residual_layers.append(nn.Linear(input_dim, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        current = x
        previous_x = x
        layer_idx = 0

        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear) and layer_idx < len(self.residual_layers) and self.use_residual:
                # Apply residual connection
                residual = self.residual_layers[layer_idx](previous_x)
                current = layer(current)

                # Add residual if dimensions match
                if residual.shape == current.shape:
                    current = current + residual

                layer_idx += 1
            else:
                current = layer(current)

        return current


class NeuralWordleModel(WordleMLModel):
    """Neural network implementation of WORDLE ML model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        device: str | None = None
    ) -> None:
        """Initialize neural WORDLE model.

        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            epochs: Number of training epochs
            device: Device to use ('cpu' or 'cuda')
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize network
        self.network = DeepWordleNet(
            input_dim=input_dim,
            hidden_dims=hidden_dims
        ).to(self.device)

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self._is_trained = False

        logger.debug(f"NeuralWordleModel initialized with {sum(p.numel() for p in self.network.parameters())} parameters")

    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """Train the neural network model.

        Args:
            features: Feature matrix of shape (n_samples, n_features)
            targets: Target values of shape (n_samples,)

        Raises:
            MLModelError: If training fails
        """
        logger.info(f"Starting neural network training with {len(features)} samples")

        try:
            # Validate input dimensions
            if features.shape[1] != self.input_dim:
                raise MLModelError(f"Feature dimension mismatch: expected {self.input_dim}, got {features.shape[1]}")

            # Create dataset and data loader
            dataset = WordleDataset(features, targets)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0  # Avoid multiprocessing issues
            )

            # Training loop
            self.network.train()
            best_loss = float('inf')
            patience = 10
            no_improve_count = 0

            for epoch in range(self.epochs):
                epoch_loss = 0.0
                num_batches = 0

                for batch_features, batch_targets in dataloader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device).unsqueeze(1)

                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.network(batch_features)
                    loss = self.criterion(outputs, batch_targets)

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                avg_loss = epoch_loss / num_batches

                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

                if (epoch + 1) % 20 == 0:
                    logger.debug(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")

            self._is_trained = True
            logger.info(f"Neural network training completed. Final loss: {best_loss:.6f}")

        except Exception as e:
            logger.error(f"Neural network training failed: {e}")
            raise MLModelError(f"Training failed: {e}") from e

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions using the neural network.

        Args:
            features: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples,)

        Raises:
            MLModelError: If prediction fails or model not trained
        """
        if not self._is_trained:
            raise MLModelError("Model must be trained before making predictions")

        try:
            self.network.eval()

            # Convert to tensor
            features_tensor = torch.FloatTensor(features).to(self.device)

            with torch.no_grad():
                outputs = self.network(features_tensor)
                predictions = outputs.cpu().numpy().flatten()

            return predictions

        except Exception as e:
            logger.error(f"Neural network prediction failed: {e}")
            raise MLModelError(f"Prediction failed: {e}") from e

    def save_model(self, filepath: str) -> None:
        """Save neural network model to file.

        Args:
            filepath: Path to save model

        Raises:
            MLModelError: If saving fails
        """
        if not self._is_trained:
            raise MLModelError("Cannot save untrained model")

        try:
            model_state = {
                'network_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'learning_rate': self.learning_rate,
                'is_trained': self._is_trained,
                'model_type': 'NeuralWordleModel'
            }

            torch.save(model_state, filepath)
            logger.info(f"Neural network model saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save neural network model: {e}")
            raise MLModelError(f"Failed to save model: {e}") from e

    def load_model(self, filepath: str) -> None:
        """Load neural network model from file.

        Args:
            filepath: Path to load model from

        Raises:
            MLModelError: If loading fails
        """
        try:
            model_state = torch.load(filepath, map_location=self.device)

            if model_state['model_type'] != 'NeuralWordleModel':
                raise MLModelError("Model type mismatch: expected NeuralWordleModel")

            # Recreate network with saved parameters
            self.input_dim = model_state['input_dim']
            self.hidden_dims = model_state['hidden_dims']
            self.learning_rate = model_state['learning_rate']

            self.network = DeepWordleNet(
                input_dim=self.input_dim,
                hidden_dims=self.hidden_dims
            ).to(self.device)

            # Load state
            self.network.load_state_dict(model_state['network_state_dict'])
            self.optimizer.load_state_dict(model_state['optimizer_state_dict'])
            self._is_trained = model_state['is_trained']

            logger.info(f"Neural network model loaded from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load neural network model: {e}")
            raise MLModelError(f"Failed to load model: {e}") from e

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the neural network model.

        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)

        return {
            'model_type': 'NeuralWordleModel',
            'is_trained': self._is_trained,
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': str(self.device),
            'description': 'Deep neural network for WORDLE guess prediction with residual connections'
        }
