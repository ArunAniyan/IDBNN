# ------------------------------------------------------Invertable DBNN -------------------------
import logging
from typing import Dict

import torch
from tqdm import tqdm


class InvertibleDBNN(torch.nn.Module):
    """Enhanced Invertible Difference Boosting Neural Network implementation with proper gradient tracking"""

    def __init__(
        self,
        forward_model: "DBNN",
        feature_dims: int,
        reconstruction_weight: float = 0.5,
        feedback_strength: float = 0.3,
        debug: bool = False,
    ):
        """
        Initialize the invertible DBNN.

        Args:
            forward_model: The forward DBNN model
            feature_dims: Number of input feature dimensions
            reconstruction_weight: Weight for reconstruction loss (0-1)
            feedback_strength: Strength of reconstruction feedback (0-1)
            debug: Enable debug logging
        """
        super().__init__()
        self.forward_model = forward_model
        self.device = forward_model.device
        self.feature_dims = feature_dims
        self.reconstruction_weight = reconstruction_weight
        self.feedback_strength = feedback_strength

        # Enable logging if debug is True
        self.debug = debug
        if debug:
            logging.basicConfig(level=logging.DEBUG)
            self.logger = logging.getLogger(__name__)

        # Initialize model components
        self.n_classes = len(self.forward_model.label_encoder.classes_)
        self.inverse_likelihood_params = None
        self.inverse_feature_pairs = None

        # Feature scaling parameters as buffers
        self.register_buffer("min_vals", None)
        self.register_buffer("max_vals", None)
        self.register_buffer("scale_factors", None)

        # Metrics tracking
        self.metrics = {
            "reconstruction_errors": [],
            "forward_errors": [],
            "total_losses": [],
            "accuracies": [],
        }

        # Initialize all components
        self._initialize_inverse_components()

    def save_inverse_model(self, custom_path: str = None) -> bool:
        try:
            save_dir = custom_path or os.path.join(
                "Model", f"Best_inverse_{self.forward_model.dataset_name}"
            )
            os.makedirs(save_dir, exist_ok=True)

            # Save model state
            model_state = {
                "weight_linear": self.weight_linear.data,
                "weight_nonlinear": self.weight_nonlinear.data,
                "bias_linear": self.bias_linear.data,
                "bias_nonlinear": self.bias_nonlinear.data,
                "feature_attention": self.feature_attention.data,
                "layer_norm": self.layer_norm.state_dict(),
                "metrics": self.metrics,
                "feature_dims": self.feature_dims,
                "n_classes": self.n_classes,
                "reconstruction_weight": self.reconstruction_weight,
                "feedback_strength": self.feedback_strength,
            }

            # Save scale parameters if they exist
            for param in [
                "min_vals",
                "max_vals",
                "scale_factors",
                "inverse_feature_pairs",
            ]:
                if hasattr(self, param):
                    model_state[param] = getattr(self, param)

            model_path = os.path.join(save_dir, "inverse_model.pt")
            torch.save(model_state, model_path)

            # Save config
            config = {
                "feature_dims": self.feature_dims,
                "reconstruction_weight": float(self.reconstruction_weight),
                "feedback_strength": float(self.feedback_strength),
                "n_classes": int(self.n_classes),
                "device": str(self.device),
            }

            config_path = os.path.join(save_dir, "inverse_config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)

            print(f"Saved inverse model to {save_dir}")
            return True

        except Exception as e:
            print(f"Error saving inverse model: {str(e)}")
            traceback.print_exc()
            return False

    def load_inverse_model(self, custom_path: str = None) -> bool:
        try:
            load_dir = custom_path or os.path.join(
                "Model", f"Best_inverse_{self.forward_model.dataset_name}"
            )
            model_path = os.path.join(load_dir, "inverse_model.pt")
            config_path = os.path.join(load_dir, "inverse_config.json")

            if not (os.path.exists(model_path) and os.path.exists(config_path)):
                print(f"No saved inverse model found at {load_dir}")
                return False

            model_state = torch.load(
                model_path, map_location=self.device, weights_only=True
            )

            with open(config_path) as f:
                config = json.load(f)

            if (
                config["feature_dims"] != self.feature_dims
                or config["n_classes"] != self.n_classes
            ):
                raise ValueError("Model architecture mismatch")

            # Load parameters
            self.weight_linear.data = model_state["weight_linear"]
            self.weight_nonlinear.data = model_state["weight_nonlinear"]
            self.bias_linear.data = model_state["bias_linear"]
            self.bias_nonlinear.data = model_state["bias_nonlinear"]
            self.feature_attention.data = model_state["feature_attention"]
            self.layer_norm.load_state_dict(model_state["layer_norm"])

            # Safely update or register buffers
            for param in [
                "min_vals",
                "max_vals",
                "scale_factors",
                "inverse_feature_pairs",
            ]:
                if param in model_state:
                    buffer_data = model_state[param]
                    if buffer_data is not None:
                        if hasattr(self, param) and getattr(self, param) is not None:
                            getattr(self, param).copy_(buffer_data)
                        else:
                            self.register_buffer(param, buffer_data)

            # Restore other attributes
            self.metrics = model_state.get("metrics", {})
            self.reconstruction_weight = model_state.get("reconstruction_weight", 0.5)
            self.feedback_strength = model_state.get("feedback_strength", 0.3)

            print(f"Loaded inverse model from {load_dir}")
            return True

        except Exception as e:
            print(f"Error loading inverse model: {str(e)}")
            traceback.print_exc()
            return False

    def _initialize_inverse_components(self):
        """Initialize inverse model parameters with proper buffer handling"""
        try:
            # Initialize feature pairs
            class_indices = torch.arange(self.n_classes, device=self.device)
            feature_indices = torch.arange(self.feature_dims, device=self.device)
            feature_pairs = torch.cartesian_prod(class_indices, feature_indices)

            # Safely register buffer
            if not hasattr(self, "inverse_feature_pairs"):
                self.register_buffer("inverse_feature_pairs", feature_pairs)
            else:
                self.inverse_feature_pairs = feature_pairs

            # Number of pairs
            n_pairs = len(feature_pairs)

            # Initialize weights as nn.Parameters
            self.weight_linear = torch.nn.Parameter(
                torch.empty((n_pairs, self.feature_dims), device=self.device),
                requires_grad=True,
            )
            self.weight_nonlinear = torch.nn.Parameter(
                torch.empty((n_pairs, self.feature_dims), device=self.device),
                requires_grad=True,
            )

            # Initialize with proper scaling
            torch.nn.init.xavier_uniform_(self.weight_linear)
            torch.nn.init.kaiming_normal_(self.weight_nonlinear)

            # Initialize biases as nn.Parameters
            self.bias_linear = torch.nn.Parameter(
                torch.zeros(self.feature_dims, device=self.device), requires_grad=True
            )
            self.bias_nonlinear = torch.nn.Parameter(
                torch.zeros(self.feature_dims, device=self.device), requires_grad=True
            )

            # Initialize layer normalization
            self.layer_norm = torch.nn.LayerNorm(self.feature_dims).to(self.device)

            # Initialize feature attention
            self.feature_attention = torch.nn.Parameter(
                torch.ones(self.feature_dims, device=self.device), requires_grad=True
            )

            # Safely register scaling buffers
            for name in ["min_vals", "max_vals", "scale_factors"]:
                if not hasattr(self, name):
                    self.register_buffer(name, None)

            if self.debug:
                self.logger.debug(f"Initialized inverse components:")
                self.logger.debug(
                    f"- Feature pairs shape: {self.inverse_feature_pairs.shape}"
                )
                self.logger.debug(f"- Linear weights shape: {self.weight_linear.shape}")
                self.logger.debug(
                    f"- Nonlinear weights shape: {self.weight_nonlinear.shape}"
                )

        except Exception as e:
            raise RuntimeError(f"Failed to initialize inverse components: {str(e)}")

    def _compute_feature_scaling(self, features: torch.Tensor):
        """Compute feature scaling parameters for consistent reconstruction"""
        with torch.no_grad():
            self.min_vals = features.min(dim=0)[0]
            self.max_vals = features.max(dim=0)[0]
            self.scale_factors = self.max_vals - self.min_vals
            self.scale_factors[self.scale_factors == 0] = 1.0

    def _scale_features(self, features: torch.Tensor) -> torch.Tensor:
        """Scale features to [0,1] range"""
        return (features - self.min_vals) / self.scale_factors

    def _unscale_features(self, scaled_features: torch.Tensor) -> torch.Tensor:
        """Convert scaled features back to original range"""
        return (scaled_features * self.scale_factors) + self.min_vals

    def _compute_inverse_posterior(self, class_probs: torch.Tensor) -> torch.Tensor:
        """Enhanced inverse posterior computation with improved stability"""
        batch_size = class_probs.shape[0]
        class_probs = class_probs.to(dtype=self.weight_linear.dtype)

        reconstructed_features = torch.zeros(
            (batch_size, self.feature_dims),
            device=self.device,
            dtype=self.weight_linear.dtype,
        )

        # Apply attention mechanism
        attention_weights = torch.softmax(self.feature_attention, dim=0)

        # Compute linear and nonlinear transformations
        linear_features = torch.zeros_like(reconstructed_features)
        nonlinear_features = torch.zeros_like(reconstructed_features)

        for feat_idx in range(self.feature_dims):
            # Get relevant pairs for this feature
            relevant_pairs = torch.where(self.inverse_feature_pairs[:, 1] == feat_idx)[
                0
            ]

            # Get class contributions
            class_contributions = class_probs[
                :, self.inverse_feature_pairs[relevant_pairs, 0]
            ]

            # Linear transformation
            linear_weights = self.weight_linear[relevant_pairs, feat_idx]
            linear_features[:, feat_idx] = torch.mm(
                class_contributions, linear_weights.unsqueeze(1)
            ).squeeze()

            # Nonlinear transformation with tanh activation
            nonlinear_weights = self.weight_nonlinear[relevant_pairs, feat_idx]
            nonlinear_features[:, feat_idx] = torch.tanh(
                torch.mm(class_contributions, nonlinear_weights.unsqueeze(1)).squeeze()
            )

        # Combine transformations with attention
        reconstructed_features = (
            attention_weights * linear_features
            + (1 - attention_weights) * nonlinear_features
        )

        # Add biases
        reconstructed_features += self.bias_linear + self.bias_nonlinear

        # Apply layer normalization
        reconstructed_features = self.layer_norm(reconstructed_features)

        return reconstructed_features

    def _compute_reconstruction_loss(
        self,
        original_features: torch.Tensor,
        reconstructed_features: torch.Tensor,
        class_probs: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Enhanced reconstruction loss with multiple components"""
        # Scale features
        orig_scaled = self._scale_features(original_features)
        recon_scaled = self._scale_features(reconstructed_features)

        # MSE loss with feature-wise weighting
        mse_loss = torch.mean((orig_scaled - recon_scaled) ** 2, dim=1)

        # Feature correlation loss
        orig_centered = orig_scaled - orig_scaled.mean(dim=0, keepdim=True)
        recon_centered = recon_scaled - recon_scaled.mean(dim=0, keepdim=True)

        corr_loss = -torch.sum(orig_centered * recon_centered, dim=1) / (
            torch.norm(orig_centered, dim=1) * torch.norm(recon_centered, dim=1) + 1e-8
        )

        # Distribution matching loss using KL divergence
        orig_dist = torch.distributions.Normal(
            orig_scaled.mean(dim=0), orig_scaled.std(dim=0) + 1e-8
        )
        recon_dist = torch.distributions.Normal(
            recon_scaled.mean(dim=0), recon_scaled.std(dim=0) + 1e-8
        )
        dist_loss = torch.distributions.kl_divergence(orig_dist, recon_dist).mean()

        # Combine losses with learned weights
        combined_loss = mse_loss + 0.1 * corr_loss + 0.01 * dist_loss

        if reduction == "mean":
            return combined_loss.mean()
        return combined_loss

    def train(self, X_train, y_train, X_test, y_test, batch_size=32):
        """Complete training with optimized test evaluation"""
        print("\nStarting training...")
        n_samples = len(X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size

        # Initialize tracking metrics
        error_rates = []
        best_train_accuracy = 0.0
        best_test_accuracy = 0.0
        patience_counter = 0
        plateau_counter = 0
        min_improvement = 0.001
        patience = 5 if self.in_adaptive_fit else 100
        max_plateau = 5
        prev_accuracy = 0.0

        # Main training loop with progress tracking
        with tqdm(total=self.max_epochs, desc="Training epochs") as epoch_pbar:
            for epoch in range(self.max_epochs):
                # Train on all batches
                failed_cases = []
                n_errors = 0

                # Process training batches
                with tqdm(
                    total=n_batches, desc=f"Training batches", leave=False
                ) as batch_pbar:
                    for i in range(0, n_samples, batch_size):
                        batch_end = min(i + batch_size, n_samples)
                        batch_X = X_train[i:batch_end]
                        batch_y = y_train[i:batch_end]

                        # Forward pass and error collection
                        posteriors = self._compute_batch_posterior(batch_X)[0]
                        predictions = torch.argmax(posteriors, dim=1)
                        errors = predictions != batch_y
                        n_errors += errors.sum().item()

                        # Collect failed cases for weight updates
                        if errors.any():
                            fail_idx = torch.where(errors)[0]
                            for idx in fail_idx:
                                failed_cases.append(
                                    (
                                        batch_X[idx],
                                        batch_y[idx].item(),
                                        posteriors[idx].cpu().numpy(),
                                    )
                                )
                        batch_pbar.update(1)

                # Update weights after processing all batches
                if failed_cases:
                    self._update_priors_parallel(failed_cases, batch_size)

                # Calculate training metrics
                train_error_rate = n_errors / n_samples
                train_accuracy = 1 - train_error_rate
                error_rates.append(train_error_rate)

                # Evaluate on test set once per epoch
                if X_test is not None and y_test is not None:
                    test_predictions = self.predict(X_test, batch_size=batch_size)
                    test_accuracy = (
                        (test_predictions == y_test.cpu()).float().mean().item()
                    )

                    # Print confusion matrix only for best test performance
                    if test_accuracy > best_test_accuracy:
                        best_test_accuracy = test_accuracy
                        print("\nTest Set Performance:")
                        y_test_labels = self.label_encoder.inverse_transform(
                            y_test.cpu().numpy()
                        )
                        test_pred_labels = self.label_encoder.inverse_transform(
                            test_predictions.cpu().numpy()
                        )
                        self.print_colored_confusion_matrix(
                            y_test_labels, test_pred_labels
                        )

                # Update progress bar with metrics
                epoch_pbar.set_postfix(
                    {
                        "train_acc": f"{train_accuracy:.4f}",
                        "best_train": f"{best_train_accuracy:.4f}",
                        "test_acc": f"{test_accuracy:.4f}",
                        "best_test": f"{best_test_accuracy:.4f}",
                    }
                )
                epoch_pbar.update(1)

                # Check improvement and update tracking
                accuracy_improvement = train_accuracy - prev_accuracy
                if accuracy_improvement <= min_improvement:
                    plateau_counter += 1
                else:
                    plateau_counter = 0

                if train_accuracy > best_train_accuracy + min_improvement:
                    best_train_accuracy = train_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Save best model
                if train_error_rate <= self.best_error:
                    self.best_error = train_error_rate
                    self.best_W = self.current_W.clone()
                    self._save_best_weights()

                # Early stopping checks
                if train_accuracy == 1.0:
                    print("\nReached 100% training accuracy")
                    break

                if patience_counter >= patience:
                    print(f"\nNo improvement for {patience} epochs")
                    break

                if plateau_counter >= max_plateau:
                    print(f"\nAccuracy plateaued for {max_plateau} epochs")
                    break

                prev_accuracy = train_accuracy

            self._save_model_components()
            return self.current_W.cpu(), error_rates

    def reconstruct_features(self, class_probs: torch.Tensor) -> torch.Tensor:
        """Reconstruct input features from class probabilities with dtype handling"""
        with torch.no_grad():
            # Ensure consistent dtype
            class_probs = class_probs.to(dtype=torch.float32)
            reconstructed = self._compute_inverse_posterior(class_probs)

            if hasattr(self, "min_vals") and self.min_vals is not None:
                reconstructed = self._unscale_features(reconstructed)
                # Ensure output matches input dtype
                return reconstructed.to(dtype=self.weight_linear.dtype)
            return reconstructed.to(dtype=self.weight_linear.dtype)

    def evaluate(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            features: Input features
            labels: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        with torch.no_grad():
            # Forward pass
            if self.forward_model.model_type == "Histogram":
                class_probs, _ = self.forward_model._compute_batch_posterior(features)
            else:
                class_probs, _ = self.forward_model._compute_batch_posterior_std(
                    features
                )

            # Get predictions and convert to numpy
            predictions = torch.argmax(class_probs, dim=1)
            predictions_np = predictions.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Convert to original class labels
            true_labels = self.forward_model.label_encoder.inverse_transform(labels_np)
            pred_labels = self.forward_model.label_encoder.inverse_transform(
                predictions_np
            )

            # Compute classification report and confusion matrix
            from sklearn.metrics import classification_report, confusion_matrix

            class_report = classification_report(true_labels, pred_labels)
            conf_matrix = confusion_matrix(true_labels, pred_labels)

            # Calculate test accuracy
            test_accuracy = (predictions == labels).float().mean().item()

            # Get training error rates from metrics history
            error_rates = self.metrics.get("forward_errors", [])
            if not error_rates and hasattr(self.forward_model, "error_rates"):
                error_rates = self.forward_model.error_rates

            # Get reconstruction metrics
            reconstructed_features = self.reconstruct_features(class_probs)
            reconstruction_loss = self._compute_reconstruction_loss(
                features, reconstructed_features, reduction="mean"
            ).item()

            # Prepare results dictionary matching expected format
            results = {
                "classification_report": class_report,
                "confusion_matrix": conf_matrix,
                "error_rates": error_rates,
                "test_accuracy": test_accuracy,
                "reconstruction_loss": reconstruction_loss,
            }

            # Format results as string for display/saving
            formatted_output = (
                f"Results for Dataset: {self.forward_model.dataset_name}\n\n"
            )
            formatted_output += f"Classification Report:\n{class_report}\n\n"
            formatted_output += "Confusion Matrix:\n"
            formatted_output += "\n".join(
                ["\t".join(map(str, row)) for row in conf_matrix]
            )
            formatted_output += "\n\nError Rates:\n"

            if error_rates:
                formatted_output += "\n".join(
                    [f"Epoch {i+1}: {rate:.4f}" for i, rate in enumerate(error_rates)]
                )
            else:
                formatted_output += "N/A"

            formatted_output += f"\n\nTest Accuracy: {test_accuracy:.4f}\n"

            # Store formatted output in results
            results["formatted_output"] = formatted_output

            return results
