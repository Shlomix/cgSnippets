import numpy as np

def generate_classification_data(n_samples=1000, n_features=20, n_informative=2, n_redundant=2, n_classes=2, random_state=None):
    np.random.seed(random_state)
    
    # Generate informative features
    X_informative = np.random.randn(n_samples, n_informative)
    
    # Generate redundant features as linear combinations of informative features
    X_redundant = np.dot(X_informative, np.random.randn(n_informative, n_redundant))
    
    # Generate noise features
    n_noise = n_features - n_informative - n_redundant
    X_noise = np.random.randn(n_samples, n_noise)
    
    # Combine all features
    X = np.hstack((X_informative, X_redundant, X_noise))
    
    # Generate labels
    y = np.random.randint(n_classes, size=n_samples)
    
    return X, y

# Example usage
X, y = generate_classification_data(n_samples=1000, n_features=20, n_informative=5, n_redundant=3, n_classes=2, random_state=42)
print(X.shape, y.shape)
