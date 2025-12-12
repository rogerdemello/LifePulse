"""
ONNX-based inference utilities for models
Uses joblib-loaded sklearn models with efficient numpy operations
"""
import joblib
import numpy as np
import os

class ONNXModelWrapper:
    """Wrapper for sklearn models to provide consistent inference interface"""
    
    def __init__(self, model_dir, model_prefix):
        """
        Load model, scaler, and features from directory
        
        Args:
            model_dir: Directory containing model files
            model_prefix: Prefix for model files (e.g., 'heart_disease', 'sleep')
        """
        self.model_dir = model_dir
        self.model_prefix = model_prefix
        
        # Load model components
        # Heart model uses heart_disease_model but heart_scaler/heart_features
        model_path = os.path.join(model_dir, f'{model_prefix}_model_improved.pkl')
        if not os.path.exists(model_path) and model_prefix == 'heart':
            model_path = os.path.join(model_dir, 'heart_disease_model_improved.pkl')
        
        scaler_path = os.path.join(model_dir, f'{model_prefix}_scaler_improved.pkl')
        features_path = os.path.join(model_dir, f'{model_prefix}_features_improved.pkl')
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.features = joblib.load(features_path)
        
        print(f"✓ Loaded {model_prefix} model with {len(self.features)} features")
    
    def predict(self, input_data):
        """
        Run inference on input data
        
        Args:
            input_data: dict or numpy array of features
        
        Returns:
            predictions (numpy array)
        """
        # Convert dict to numpy array if needed
        if isinstance(input_data, dict):
            X = np.array([[input_data.get(f, 0) for f in self.features]])
        else:
            X = np.array(input_data).reshape(1, -1)
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, input_data):
        """
        Get prediction probabilities (for classifiers)
        
        Args:
            input_data: dict or numpy array of features
        
        Returns:
            probability array
        """
        # Convert dict to numpy array if needed
        if isinstance(input_data, dict):
            X = np.array([[input_data.get(f, 0) for f in self.features]])
        else:
            X = np.array(input_data).reshape(1, -1)
        
        # Scale and predict probabilities
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            # For regressors, return predictions as-is
            return self.model.predict(X_scaled)


# Global model cache
_model_cache = {}

def get_model(model_name):
    """
    Get or load a model from cache
    
    Args:
        model_name: 'heart', 'sleep', 'migraine', or 'health_score'
    
    Returns:
        ONNXModelWrapper instance
    """
    if model_name in _model_cache:
        return _model_cache[model_name]
    
    # Map model names to directories and prefixes
    # Note: heart uses mixed naming (heart_disease for model, heart for scaler/features)
    model_config = {
        'heart': ('app/models/heart', 'heart'),
        'sleep': ('app/models/sleep', 'sleep'),
        'migraine': ('app/models/migraine', 'migraine'),
        'health_score': ('app/models/health_score', 'health_score')
    }
    
    if model_name not in model_config:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_dir, prefix = model_config[model_name]
    model = ONNXModelWrapper(model_dir, prefix)
    _model_cache[model_name] = model
    
    return model
