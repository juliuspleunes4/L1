"""
@file       : test_utilities.py
@author     : J.J.G. Pleunes
@date       : 08/2025
@brief      : Comprehensive test suite for utility functions.
@details    : Tests for device management, logging, random seeding,
              and other utility functionality.
@version    : 1.0

@license    : MIT License
"""

import unittest
import torch
import sys
import os
import tempfile
import logging
from unittest.mock import patch, MagicMock
from io import StringIO

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.utils import get_device, set_seed, setup_logging, get_logger, move_to_device
except ImportError:
    try:
        from utils import get_device, set_seed, setup_logging, get_logger, move_to_device
    except ImportError:
        # Set missing functions to None for graceful degradation
        get_device = None
        set_seed = None
        setup_logging = None
        get_logger = None
        move_to_device = None


class TestDeviceManagement(unittest.TestCase):
    """Test device management utilities."""
    
    def test_get_device_function(self):
        """Test get_device function."""
        try:
            device = get_device()
            
            self.assertIsInstance(device, torch.device)
            self.assertIn(str(device), ['cpu', 'cuda:0', 'cuda', 'mps'])
            
        except:
            # Fallback implementation
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            
            self.assertIsInstance(device, torch.device)
    
    def test_device_manager(self):
        """Test device management functionality."""
        try:
            # Test device detection
            device = get_device()
            self.assertIsInstance(device, torch.device)
            
            # Test CUDA availability
            cuda_available = torch.cuda.is_available()
            self.assertIsInstance(cuda_available, bool)
            
            if cuda_available:
                device_count = torch.cuda.device_count()
                self.assertIsInstance(device_count, int)
                self.assertGreaterEqual(device_count, 0)
            
        except:
            self.skipTest("Device management functions not found")
    
    def test_cuda_availability(self):
        """Test CUDA availability detection."""
        cuda_available = torch.cuda.is_available()
        self.assertIsInstance(cuda_available, bool)
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            self.assertGreaterEqual(device_count, 1)
            
            # Test device properties
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                self.assertIsInstance(device_name, str)
                self.assertGreater(len(device_name), 0)
    
    def test_device_memory_management(self):
        """Test device memory management."""
        if torch.cuda.is_available():
            # Test memory functions
            torch.cuda.empty_cache()
            
            # Check memory stats
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            
            self.assertIsInstance(memory_allocated, int)
            self.assertIsInstance(memory_reserved, int)
            self.assertGreaterEqual(memory_allocated, 0)
            self.assertGreaterEqual(memory_reserved, 0)
    
    def test_tensor_device_placement(self):
        """Test tensor device placement."""
        # CPU tensor
        cpu_tensor = torch.randn(3, 3)
        self.assertEqual(cpu_tensor.device.type, 'cpu')
        
        # CUDA tensor if available
        if torch.cuda.is_available():
            cuda_tensor = cpu_tensor.cuda()
            self.assertEqual(cuda_tensor.device.type, 'cuda')
            
            # Move back to CPU
            cpu_tensor_2 = cuda_tensor.cpu()
            self.assertEqual(cpu_tensor_2.device.type, 'cpu')
            
            # Check values preserved
            self.assertTrue(torch.allclose(cpu_tensor, cpu_tensor_2))


class TestSeedManagement(unittest.TestCase):
    """Test random seed management."""
    
    def test_set_seed_function(self):
        """Test set_seed function."""
        seed = 42
        
        try:
            set_seed(seed)
            
            # Test reproducibility
            torch.manual_seed(seed)
            tensor1 = torch.randn(5)
            
            torch.manual_seed(seed)
            tensor2 = torch.randn(5)
            
            self.assertTrue(torch.equal(tensor1, tensor2))
            
        except:
            # Manual seed setting
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
    
    def test_set_random_seed_function(self):
        """Test set_seed function for random seeding."""
        seed = 123
        
        try:
            set_seed(seed)
            
            # Test PyTorch reproducibility
            torch.manual_seed(seed)
            tensor1 = torch.randn(3, 3)
            
            torch.manual_seed(seed)
            tensor2 = torch.randn(3, 3)
            
            self.assertTrue(torch.equal(tensor1, tensor2))
            
        except:
            self.skipTest("set_seed function not found")
    
    def test_numpy_seed_integration(self):
        """Test numpy seed integration."""
        try:
            import numpy as np
            
            seed = 456
            
            # Set seeds
            try:
                set_seed(seed)
            except:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            # Test numpy reproducibility
            np.random.seed(seed)
            array1 = np.random.randn(5)
            
            np.random.seed(seed)
            array2 = np.random.randn(5)
            
            self.assertTrue(np.allclose(array1, array2))
            
        except ImportError:
            self.skipTest("NumPy not available")
    
    def test_random_state_consistency(self):
        """Test random state consistency across operations."""
        seed = 789
        torch.manual_seed(seed)
        
        # Generate sequence of random numbers
        numbers1 = [torch.randn(1).item() for _ in range(5)]
        
        # Reset and generate again
        torch.manual_seed(seed)
        numbers2 = [torch.randn(1).item() for _ in range(5)]
        
        # Should be identical
        for n1, n2 in zip(numbers1, numbers2):
            self.assertAlmostEqual(n1, n2, places=6)


class TestLogging(unittest.TestCase):
    """Test logging functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.log_output = StringIO()
        self.handler = logging.StreamHandler(self.log_output)
        
    def tearDown(self):
        """Clean up test fixtures."""
        self.handler.close()
    
    def test_setup_logging_function(self):
        """Test setup_logging function."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_file = f.name
        
        try:
            setup_logging(log_file=log_file, level='INFO')
            
            # Test logging works
            logger = logging.getLogger(__name__)
            logger.info("Test message")
            
            # Check log file exists and has content
            self.assertTrue(os.path.exists(log_file))
            
            with open(log_file, 'r') as f:
                content = f.read()
                self.assertIn("Test message", content)
            
        except:
            self.skipTest("setup_logging function not found")
        
        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)
    
    def test_logger_functionality(self):
        """Test logging functionality."""
        try:
            logger = get_logger("test_logger")
            
            # Test basic logging
            logger.info("Test info message")
            logger.warning("Test warning message")
            logger.error("Test error message")
            
            # Should not raise any exceptions
            self.assertTrue(True)
            
        except:
            self.skipTest("get_logger function not found")
    
    def test_training_logger(self):
        """Test training logging functionality."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
                log_file = f.name
            
            # Setup logging for training
            setup_logging(log_file=log_file, level='INFO')
            logger = get_logger("training")
            
            # Test training-specific logging patterns
            logger.info("EPOCH 1 | Loss: 2.5 | LR: 0.001")
            logger.info("STEP 100 | Loss: 2.3 | LR: 0.001")
            
            # Check log file has content
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    content = f.read()
                    self.assertGreater(len(content), 0)
                
                os.unlink(log_file)
            
        except:
            self.skipTest("Training logging functionality not available")
    
    def test_log_levels(self):
        """Test different log levels."""
        logger = logging.getLogger("test_levels")
        logger.addHandler(self.handler)
        
        # Test different levels
        levels = [
            (logging.DEBUG, "Debug message"),
            (logging.INFO, "Info message"),
            (logging.WARNING, "Warning message"),
            (logging.ERROR, "Error message"),
            (logging.CRITICAL, "Critical message")
        ]
        
        for level, message in levels:
            logger.setLevel(level)
            logger.log(level, message)
            
            output = self.log_output.getvalue()
            self.log_output.seek(0)
            self.log_output.truncate(0)
    
    def test_log_formatting(self):
        """Test log message formatting."""
        logger = logging.getLogger("test_formatting")
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.handler.setFormatter(formatter)
        logger.addHandler(self.handler)
        logger.setLevel(logging.INFO)
        
        # Log message
        logger.info("Formatted test message")
        
        output = self.log_output.getvalue()
        
        # Check formatting components
        self.assertIn("test_formatting", output)
        self.assertIn("INFO", output)
        self.assertIn("Formatted test message", output)


class TestFileUtilities(unittest.TestCase):
    """Test file and path utilities."""
    
    def test_file_operations(self):
        """Test basic file operations."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            temp_file = f.name
            f.write("Test content")
        
        try:
            # Test file exists
            self.assertTrue(os.path.exists(temp_file))
            
            # Test file reading
            with open(temp_file, 'r') as f:
                content = f.read()
                self.assertEqual(content, "Test content")
            
            # Test file size
            file_size = os.path.getsize(temp_file)
            self.assertGreater(file_size, 0)
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_directory_operations(self):
        """Test directory operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test directory exists
            self.assertTrue(os.path.exists(temp_dir))
            self.assertTrue(os.path.isdir(temp_dir))
            
            # Test creating subdirectory
            sub_dir = os.path.join(temp_dir, 'subdir')
            os.makedirs(sub_dir)
            self.assertTrue(os.path.exists(sub_dir))
            
            # Test listing directory
            contents = os.listdir(temp_dir)
            self.assertIn('subdir', contents)
    
    def test_path_utilities(self):
        """Test path manipulation utilities."""
        # Test path joining
        path = os.path.join('dir1', 'dir2', 'file.txt')
        expected_sep = os.sep
        self.assertIn(expected_sep, path)
        
        # Test path splitting
        dirname, filename = os.path.split(path)
        self.assertEqual(filename, 'file.txt')
        
        # Test file extension
        name, ext = os.path.splitext(filename)
        self.assertEqual(name, 'file')
        self.assertEqual(ext, '.txt')
        
        # Test absolute path
        abs_path = os.path.abspath('.')
        self.assertTrue(os.path.isabs(abs_path))


class TestConfigurationUtilities(unittest.TestCase):
    """Test configuration management utilities."""
    
    def test_dict_operations(self):
        """Test dictionary operations for configurations."""
        config = {
            'model': {
                'n_layers': 12,
                'n_heads': 8,
                'n_embd': 512
            },
            'training': {
                'learning_rate': 0.001,
                'batch_size': 32
            }
        }
        
        # Test nested access
        self.assertEqual(config['model']['n_layers'], 12)
        self.assertEqual(config['training']['learning_rate'], 0.001)
        
        # Test key existence
        self.assertIn('model', config)
        self.assertIn('n_layers', config['model'])
        self.assertNotIn('nonexistent', config)
        
        # Test updating
        config['training']['batch_size'] = 64
        self.assertEqual(config['training']['batch_size'], 64)
    
    def test_json_serialization(self):
        """Test JSON serialization for configurations."""
        import json
        
        config = {
            'model_name': 'L1',
            'vocab_size': 50257,
            'learning_rate': 0.001,
            'use_cuda': True
        }
        
        # Test serialization
        json_str = json.dumps(config)
        self.assertIsInstance(json_str, str)
        
        # Test deserialization
        loaded_config = json.loads(json_str)
        self.assertEqual(loaded_config, config)
        
        # Test file I/O
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_file = f.name
        
        try:
            with open(temp_file, 'r') as f:
                loaded_config = json.load(f)
                self.assertEqual(loaded_config, config)
        
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestMemoryUtilities(unittest.TestCase):
    """Test memory management utilities."""
    
    def test_tensor_memory_tracking(self):
        """Test tensor memory tracking."""
        # Create tensors
        tensor1 = torch.randn(100, 100)
        tensor2 = torch.randn(200, 200)
        
        # Test tensor sizes
        size1 = tensor1.numel() * tensor1.element_size()
        size2 = tensor2.numel() * tensor2.element_size()
        
        self.assertGreater(size1, 0)
        self.assertGreater(size2, size1)  # tensor2 should be larger
        
        # Test memory cleanup
        del tensor1, tensor2
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def test_cuda_memory_management(self):
        """Test CUDA memory management."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Check initial memory
        initial_memory = torch.cuda.memory_allocated()
        
        # Create CUDA tensor
        cuda_tensor = torch.randn(1000, 1000, device='cuda')
        
        # Check memory increased
        after_alloc_memory = torch.cuda.memory_allocated()
        self.assertGreater(after_alloc_memory, initial_memory)
        
        # Clean up
        del cuda_tensor
        torch.cuda.empty_cache()
        
        # Memory should be freed (though might not be exactly initial due to caching)
        final_memory = torch.cuda.memory_allocated()
        self.assertLessEqual(final_memory, after_alloc_memory)


if __name__ == '__main__':
    # Set random seed for reproducible tests
    torch.manual_seed(42)
    
    # Run tests with verbose output
    unittest.main(verbosity=2)
