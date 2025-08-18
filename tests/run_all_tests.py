#!/usr/bin/env python3
"""
@file       : run_all_tests.py
@author     : J.J.G. Pleunes
@date       : 08/2025
@brief      : Comprehensive test runner for L1 project
@details    : Runs all test modules and provides detailed reporting
@version    : 1.0

@license    : MIT License
Copyright (c) 2025 Julius Pleunes
"""

import sys
import os
import unittest
import importlib
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_all_tests():
    """Run all test modules with comprehensive reporting"""
    
    # Test modules to run
    test_modules = [
        'test_models_comprehensive',
        'test_data_processing', 
        'test_training_pipeline',
        'test_utilities',
        'test_integration'
    ]
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    print("="*80)
    print("L1 PROJECT COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Running {len(test_modules)} test modules...")
    print()
    
    # Load and add test modules
    for module_name in test_modules:
        try:
            print(f"Loading {module_name}...")
            module = importlib.import_module(f'tests.{module_name}')
            module_suite = unittest.TestLoader().loadTestsFromModule(module)
            test_suite.addTest(module_suite)
            print(f"✓ Successfully loaded {module_name}")
        except ImportError as e:
            print(f"✗ Failed to load {module_name}: {e}")
        except Exception as e:
            print(f"✗ Error loading {module_name}: {e}")
    
    print()
    print("="*80)
    print("RUNNING TESTS")
    print("="*80)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    result = runner.run(test_suite)
    
    print()
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    print("="*80)
    
    # Return success/failure status
    return len(result.failures) == 0 and len(result.errors) == 0

def run_specific_test(test_name):
    """Run a specific test module"""
    
    print(f"Running specific test: {test_name}")
    print("="*50)
    
    try:
        module = importlib.import_module(f'tests.{test_name}')
        suite = unittest.TestLoader().loadTestsFromModule(module)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        print(f"\nTest {test_name} completed:")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        return len(result.failures) == 0 and len(result.errors) == 0
        
    except ImportError as e:
        print(f"Failed to import test module {test_name}: {e}")
        return False

def run_test_discovery():
    """Run test discovery to find and run all test files"""
    
    print("Running test discovery...")
    print("="*50)
    
    # Discover tests in the tests directory
    test_dir = Path(__file__).parent
    loader = unittest.TestLoader()
    
    try:
        suite = loader.discover(
            start_dir=str(test_dir),
            pattern='test_*.py',
            top_level_dir=str(project_root)
        )
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        print(f"\nTest discovery completed:")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        return len(result.failures) == 0 and len(result.errors) == 0
        
    except Exception as e:
        print(f"Test discovery failed: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'discover':
            success = run_test_discovery()
        else:
            # Run specific test module
            test_name = sys.argv[1]
            if not test_name.startswith('test_'):
                test_name = f'test_{test_name}'
            success = run_specific_test(test_name)
    else:
        # Run all tests
        success = run_all_tests()
    
    sys.exit(0 if success else 1)
