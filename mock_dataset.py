import torch
from torch.utils.data import Dataset

class MockDataset(Dataset):
    """Mock dataset for demonstration"""
    def __init__(self, size=100):
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate mock image and target
        image = torch.randn(3, 224, 224)
        target = {
            'boxes': torch.tensor([[10, 10, 50, 50], [60, 60, 100, 100]]),
            'labels': torch.tensor([1, 2])
        }
        return image, target