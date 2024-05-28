import torch

def generate_classification_data(n_samples=1000, n_features=20, n_informative=2, n_redundant=2, n_classes=2, random_state=None):
    if random_state is not None:
        torch.manual_seed(random_state)
    
    # Generate informative features
    X_informative = torch.randn(n_samples, n_informative)
    
    # Generate redundant features as linear combinations of informative features
    X_redundant = torch.matmul(X_informative, torch.randn(n_informative, n_redundant))
    
    # Generate noise features
    n_noise = n_features - n_informative - n_redundant
    X_noise = torch.randn(n_samples, n_noise)
    
    # Combine all features
    X = torch.cat((X_informative, X_redundant, X_noise), dim=1)
    
    # Generate labels
    y = torch.randint(n_classes, (n_samples,))
    
    return X, y

# Example usage
X, y = generate_classification_data(n_samples=1000, n_features=20, n_informative=5, n_redundant=3, n_classes=2, random_state=42)
print(X.shape, y.shape)
