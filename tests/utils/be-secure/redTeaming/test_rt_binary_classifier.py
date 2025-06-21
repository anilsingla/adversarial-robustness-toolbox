import numpy as np
import pytest
import tempfile
import os
import json
import sys
from unittest.mock import MagicMock, patch, mock_open
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D

# Add the correct path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../utils/be-secure/redTeaming'))

# Import the functions to test
from rt_binary_classifier import (
    load_classifier_model,
    prepare_data,
    create_art_classifier,
    ModelWrapper,
    run_attack,
    get_attack_configurations,
    save_results,
    visualize_results
)

class TestLoadClassifierModel:
    """Test cases for load_classifier_model function."""
    
    def test_load_nonexistent_model(self):
        """Test loading a model that doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            load_classifier_model("/nonexistent/path/model.h5")
    
    @patch('rt_binary_classifier.load_model')
    @patch('rt_binary_classifier.os.path.exists')
    def test_load_model_success(self, mock_exists, mock_load_model):
        """Test successful model loading."""
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_model.input_shape = (None, 28, 28, 1)
        mock_model.output_shape = (None, 10)
        mock_model.optimizer = 'adam'
        mock_load_model.return_value = mock_model
        
        result = load_classifier_model("/fake/path/model.h5")
        
        assert result == mock_model
        mock_load_model.assert_called_once_with("/fake/path/model.h5")
    
    @patch('rt_binary_classifier.load_model')
    @patch('rt_binary_classifier.os.path.exists')
    def test_load_model_compile_if_needed(self, mock_exists, mock_load_model):
        """Test model compilation when optimizer is None."""
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_model.input_shape = (None, 28, 28, 1)
        mock_model.output_shape = (None, 10)
        mock_model.optimizer = None
        mock_load_model.return_value = mock_model
        
        result = load_classifier_model("/fake/path/model.h5")
        
        mock_model.compile.assert_called_once()
        assert result == mock_model

class TestPrepareData:
    """Test cases for prepare_data function."""
    
    @patch('rt_binary_classifier.mnist.load_data')
    def test_prepare_data_default(self, mock_load_data):
        """Test data preparation with default parameters."""
        # Mock MNIST data
        x_train = np.random.rand(60000, 28, 28)
        y_train = np.random.randint(0, 10, 60000)
        x_test = np.random.rand(10000, 28, 28)
        y_test = np.random.randint(0, 10, 10000)
        mock_load_data.return_value = ((x_train, y_train), (x_test, y_test))
        
        x_sample, y_sample_cat, y_sample_labels = prepare_data(num_samples=50)
        
        assert x_sample.shape == (50, 28, 28, 1)
        assert y_sample_cat.shape == (50, 10)
        assert y_sample_labels.shape == (50,)
        assert np.max(x_sample) <= 1.0
        assert np.min(x_sample) >= 0.0

class TestModelWrapper:
    """Test cases for ModelWrapper class."""
    
    def test_model_wrapper_predict(self):
        """Test ModelWrapper predict method."""
        mock_model = MagicMock()
        mock_predictions = np.array([[0.1, 0.9], [0.8, 0.2]])
        mock_model.predict.return_value = mock_predictions
        
        wrapper = ModelWrapper(mock_model)
        x_test = np.random.rand(2, 28, 28, 1)
        
        result = wrapper.predict(x_test)
        
        np.testing.assert_array_equal(result, mock_predictions)
        mock_model.predict.assert_called_once_with(x_test, verbose=0)

class TestRunAttack:
    """Test cases for run_attack function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_classifier = MagicMock()
        self.mock_classifier.predict.return_value = np.array([[1, 0], [0, 1]])
        
        self.x_test = np.random.rand(2, 28, 28, 1)
        self.y_test = np.array([[1, 0], [0, 1]])
        self.y_cat = np.array([[1, 0], [0, 1]])
    
    def test_run_attack_success_with_y_cat(self):
        """Test successful attack execution with y_cat parameter."""
        class MockAttack:
            def __init__(self, estimator, **kwargs):
                self.estimator = estimator
                self.kwargs = kwargs
            
            def generate(self, x, y=None):
                return x + 0.1  # Simulate adversarial perturbation
        
        result, x_adv = run_attack(
            MockAttack, {"eps": 0.1}, self.mock_classifier, 
            self.x_test, self.y_test, "MockAttack", y_cat=self.y_cat
        )
        
        assert result['success'] is True
        assert 'accuracy' in result
        assert 'avg_perturbation' in result
        assert x_adv is not None
    
    def test_run_attack_failure(self):
        """Test attack failure handling."""
        class FailingAttack:
            def __init__(self, estimator, **kwargs):
                self.estimator = estimator
            
            def generate(self, x, y=None):
                raise Exception("Attack failed")
        
        result, x_adv = run_attack(
            FailingAttack, {}, self.mock_classifier, 
            self.x_test, self.y_test, "FailingAttack"
        )
        
        assert result['success'] is False
        assert 'error' in result
        assert x_adv is None

class TestSaveResults:
    """Test cases for save_results function."""
    
    def test_save_results_success(self):
        """Test successful results saving."""
        results = {
            'FGSM': {
                'accuracy': np.float64(0.85),
                'avg_perturbation': np.float32(0.1),
                'success': True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            metadata = {'model_path': '/fake/path', 'num_samples': 100}
            save_results(results, tmp_path, metadata)
            
            # Verify file was created and contains expected data
            assert os.path.exists(tmp_path)
            
            with open(tmp_path, 'r') as f:
                saved_data = json.load(f)
            
            assert 'timestamp' in saved_data
            assert 'metadata' in saved_data
            assert 'results' in saved_data
            assert saved_data['metadata'] == metadata
            assert 'FGSM' in saved_data['results']
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

class TestGetAttackConfigurations:
    """Test cases for get_attack_configurations function."""
    
    def test_get_fast_attacks_structure(self):
        """Test that fast attacks return proper structure."""
        attacks = get_attack_configurations('fast')
        
        assert isinstance(attacks, list)
        assert len(attacks) > 0
        
        # Check structure of each attack configuration
        for attack in attacks:
            assert len(attack) == 3  # (class, params, name)
            attack_class, params, name = attack
            assert callable(attack_class) or hasattr(attack_class, '__call__')
            assert isinstance(params, dict)
            assert isinstance(name, str)
    
    def test_get_comprehensive_attacks_structure(self):
        """Test that comprehensive attacks return proper structure."""
        attacks = get_attack_configurations('comprehensive')
        
        assert isinstance(attacks, list)
        assert len(attacks) >= len(get_attack_configurations('fast'))  # Should have at least as many as fast
        
        # Check that we have different attack types
        attack_names = [attack[2] for attack in attacks]
        assert len(set(attack_names)) == len(attack_names)  # All names should be unique
    
    def test_get_custom_attacks_structure(self):
        """Test that custom attacks return proper structure."""
        attacks = get_attack_configurations('custom')
        
        assert isinstance(attacks, list)
        assert len(attacks) >= len(get_attack_configurations('fast'))
        
        # Check for parameter variations
        attack_params = [attack[1] for attack in attacks]
        eps_values = [params.get('eps') for params in attack_params if 'eps' in params]
        assert len(set(eps_values)) > 1  # Should have different epsilon values
    
    def test_attack_configurations_have_valid_parameters(self):
        """Test that attack configurations have valid parameters."""
        for attack_type in ['fast', 'comprehensive', 'custom']:
            attacks = get_attack_configurations(attack_type)
            
            for attack_class, params, name in attacks:
                # Check that parameters are reasonable
                if 'eps' in params:
                    assert 0 < params['eps'] <= 1.0
                if 'max_iter' in params:
                    assert params['max_iter'] > 0
                if 'norm' in params:
                    assert params['norm'] in [np.inf, 1, 2] or isinstance(params['norm'], (int, float))

# Fixtures for pytest
@pytest.fixture
def temp_model_file():
    """Fixture providing a temporary model file."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        # Create a simple model and save it
        model = Sequential([
            Dense(10, activation='softmax', input_shape=(784,))
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.save(tmp_file.name)
        
        yield tmp_file.name
        
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)

class TestIntegration:
    """Integration test cases."""
    
    def create_dummy_model(self):
        """Create a dummy Keras model for testing."""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            Flatten(),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def test_model_wrapper_integration(self):
        """Test ModelWrapper with real Keras model."""
        model = self.create_dummy_model()
        wrapper = ModelWrapper(model)
        
        x_test = np.random.rand(5, 28, 28, 1)
        predictions = wrapper.predict(x_test)
        
        assert predictions.shape == (5, 10)
        assert np.all(predictions >= 0)  # Softmax outputs should be non-negative
        assert np.allclose(np.sum(predictions, axis=1), 1.0)  # Should sum to 1

if __name__ == '__main__':
    pytest.main([__file__])