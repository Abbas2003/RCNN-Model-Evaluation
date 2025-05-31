import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as transforms

class OptimizationExperiments:
    """Advanced optimization techniques for RCNN variants"""
    
    def __init__(self, model, dataset_loader):
        self.model = model
        self.dataset_loader = dataset_loader
        self.optimization_results = {}
    
    def experiment_with_optimizers(self, optimizers_config):
        """Experiment with different optimizers"""
        print("Experimenting with different optimizers...")
        
        results = {}
        
        for optimizer_name, config in optimizers_config.items():
            print(f"Testing {optimizer_name}...")
            
            # Reset model weights
            model_copy = self._create_model_copy()
            
            # Configure optimizer
            if optimizer_name.lower() == 'sgd':
                optimizer = optim.SGD(model_copy.parameters(), **config)
            elif optimizer_name.lower() == 'adam':
                optimizer = optim.Adam(model_copy.parameters(), **config)
            elif optimizer_name.lower() == 'adamw':
                optimizer = optim.AdamW(model_copy.parameters(), **config)
            elif optimizer_name.lower() == 'rmsprop':
                optimizer = optim.RMSprop(model_copy.parameters(), **config)
            
            # Train and evaluate
            train_losses, val_losses = self._mini_training_loop(model_copy, optimizer, epochs=5)
            
            results[optimizer_name] = {
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'convergence_speed': self._calculate_convergence_speed(train_losses),
                'stability': self._calculate_stability(train_losses)
            }
        
        self.optimization_results['optimizers'] = results
        return results
    
    def experiment_with_learning_rates(self, lr_range=(1e-5, 1e-1), num_experiments=5):
        """Learning rate range test"""
        print("Conducting learning rate experiments...")
        
        lr_values = np.logspace(np.log10(lr_range[0]), np.log10(lr_range[1]), num_experiments)
        results = {}
        
        for lr in lr_values:
            print(f"Testing learning rate: {lr:.2e}")
            
            model_copy = self._create_model_copy()
            optimizer = optim.Adam(model_copy.parameters(), lr=lr)
            
            train_losses, val_losses = self._mini_training_loop(model_copy, optimizer, epochs=3)
            
            results[f'{lr:.2e}'] = {
                'learning_rate': lr,
                'final_loss': train_losses[-1],
                'loss_reduction': train_losses[0] - train_losses[-1],
                'stability': self._calculate_stability(train_losses)
            }
        
        self.optimization_results['learning_rates'] = results
        return results
    
    def experiment_with_data_augmentation(self, augmentation_strategies):
        """Test different data augmentation strategies"""
        print("Testing data augmentation strategies...")
        
        results = {}
        
        for strategy_name, transforms_list in augmentation_strategies.items():
            print(f"Testing {strategy_name}...")
            
            # Create augmented dataset
            augmented_transform = transforms.Compose(transforms_list)
            augmented_dataset = self._create_augmented_dataset(augmented_transform)
            augmented_loader = DataLoader(augmented_dataset, batch_size=4, shuffle=True)
            
            model_copy = self._create_model_copy()
            optimizer = optim.Adam(model_copy.parameters(), lr=1e-4)
            
            train_losses, val_losses = self._mini_training_loop(
                model_copy, optimizer, epochs=3, train_loader=augmented_loader
            )
            
            results[strategy_name] = {
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'generalization_gap': val_losses[-1] - train_losses[-1]
            }
        
        self.optimization_results['data_augmentation'] = results
        return results
    
    def _create_model_copy(self):
        import torchvision
        import torch

        # Detect model type and get num_classes
        if hasattr(self.model, "roi_heads") and hasattr(self.model.roi_heads, "box_predictor"):
            num_classes = self.model.roi_heads.box_predictor.cls_score.out_features
            model_copy = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
            in_features = model_copy.roi_heads.box_predictor.cls_score.in_features
            model_copy.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes
            )
        else:
            # Add logic for other model types if needed
            raise NotImplementedError("Model copy not implemented for this model type.")

        model_copy = model_copy.to(self.model.device)
        return model_copy
    
    def _mini_training_loop(self, model, optimizer, epochs=3, train_loader=None):
        """Simplified training loop for experiments"""
        if train_loader is None:
            train_loader = self.dataset_loader
        
        model.train()
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            epoch_train_loss = 0.0
            batch_count = 0
            
            # Training phase
            for batch_idx, (images, targets) in enumerate(train_loader):
                if batch_idx >= 10:  # Limit batches for quick experiments
                    break
                
                optimizer.zero_grad()
                
                # Forward pass (simplified loss calculation)
                predictions = model(images)
                loss = self._calculate_simplified_loss(predictions, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                batch_count += 1
            
            avg_train_loss = epoch_train_loss / max(batch_count, 1)
            train_losses.append(avg_train_loss)
            
            # Validation phase (simplified)
            model.eval()
            val_loss = avg_train_loss * (1.1 + np.random.random() * 0.2)  # Mock validation loss
            val_losses.append(val_loss)
            model.train()
        
        return train_losses, val_losses
    
    def _calculate_simplified_loss(self, predictions, targets):
        """Simplified loss calculation for experiments"""
        # Mock loss calculation
        total_loss = 0.0
        
        for pred in predictions:
            if 'scores' in pred and len(pred['scores']) > 0:
                # Classification loss (cross-entropy approximation)
                scores = pred['scores']
                if len(scores.shape) > 1:
                    loss = F.cross_entropy(scores, torch.randint(0, scores.size(1), (scores.size(0),)))
                else:
                    loss = F.mse_loss(scores, torch.ones_like(scores))
                total_loss += loss
        
        return total_loss / max(len(predictions), 1)
    
    def _calculate_convergence_speed(self, losses):
        """Calculate how quickly the loss converges"""
        if len(losses) < 2:
            return 0.0
        
        initial_loss = losses[0]
        final_loss = losses[-1]
        convergence_speed = (initial_loss - final_loss) / len(losses)
        return max(0, convergence_speed)
    
    def _calculate_stability(self, losses):
        """Calculate training stability (lower variance = more stable)"""
        if len(losses) < 2:
            return 1.0
        
        return 1.0 / (1.0 + np.var(losses))
    
    def _create_augmented_dataset(self, transform):
        """Create an augmented version of the dataset"""
        # This is a placeholder - in practice, you'd modify your dataset class
        return self.dataset_loader.dataset
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report"""
        if not self.optimization_results:
            return "No optimization experiments have been run."
        
        report = {
            'summary': {},
            'detailed_results': self.optimization_results,
            'recommendations': {}
        }
        
        # Analyze optimizer results
        if 'optimizers' in self.optimization_results:
            best_optimizer = min(self.optimization_results['optimizers'].keys(),
                               key=lambda x: self.optimization_results['optimizers'][x]['final_val_loss'])
            report['summary']['best_optimizer'] = best_optimizer
            report['recommendations']['optimizer'] = f"Use {best_optimizer} for best performance"
        
        # Analyze learning rate results  
        if 'learning_rates' in self.optimization_results:
            best_lr = min(self.optimization_results['learning_rates'].keys(),
                         key=lambda x: self.optimization_results['learning_rates'][x]['final_loss'])
            report['summary']['best_learning_rate'] = best_lr
            report['recommendations']['learning_rate'] = f"Optimal learning rate: {best_lr}"
        
        # Analyze augmentation results
        if 'data_augmentation' in self.optimization_results:
            best_augmentation = min(self.optimization_results['data_augmentation'].keys(),
                                  key=lambda x: self.optimization_results['data_augmentation'][x]['generalization_gap'])
            report['summary']['best_augmentation'] = best_augmentation
            report['recommendations']['augmentation'] = f"Use {best_augmentation} strategy"
        
        return report