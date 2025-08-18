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
import time
from io import StringIO
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_all_tests():
    """Run all test modules with comprehensive reporting"""
    
    # Test modules to run
    test_modules = [
        ('test_models_comprehensive', 'Model Architecture & Components'),
        ('test_data_processing', 'Data Processing & Tokenization'), 
        ('test_training_pipeline', 'Training Pipeline & Configuration'),
        ('test_utilities', 'Utility Functions & Helpers'),
        ('test_integration', 'Integration & End-to-End Tests')
    ]
    
    # Create test suite
    test_suite = unittest.TestSuite()
    loaded_modules = []
    failed_modules = []
    
    print("\n" + "="*80)
    print("🚀 L1 PROJECT COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Load test modules with progress
    print(f"📦 Loading {len(test_modules)} test modules...\n")
    
    for i, (module_name, description) in enumerate(test_modules, 1):
        try:
            print(f"[{i}/{len(test_modules)}] Loading {description}...")
            module = importlib.import_module(f'tests.{module_name}')
            module_suite = unittest.TestLoader().loadTestsFromModule(module)
            test_count = module_suite.countTestCases()
            test_suite.addTest(module_suite)
            loaded_modules.append((module_name, description, test_count))
            print(f"    ✅ {module_name} ({test_count} tests)")
        except Exception as e:
            failed_modules.append((module_name, description, str(e)))
            print(f"    ❌ {module_name} - Error: {str(e)[:60]}...")
    
    total_tests = test_suite.countTestCases()
    
    print(f"\n📊 Module Loading Summary:")
    print(f"    ✅ Successfully loaded: {len(loaded_modules)} modules")
    print(f"    ❌ Failed to load: {len(failed_modules)} modules")
    print(f"    📈 Total tests discovered: {total_tests}")
    
    if failed_modules:
        print(f"\n⚠️  Failed modules:")
        for module_name, description, error in failed_modules:
            print(f"    • {description}: {error[:50]}...")
    
    print("\n" + "="*80)
    print("🧪 RUNNING TESTS")
    print("="*80)
    
    # Custom test runner with cleaner output
    stream = StringIO()
    start_time = time.time()
    
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=0,  # Reduce verbosity for cleaner output
        descriptions=False,
        failfast=False
    )
    
    # Show progress during test execution
    print("Running tests", end="", flush=True)
    
    result = runner.run(test_suite)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Clear the progress dots
    print(f"\r{'':50}", end="\r")
    
    # Calculate statistics
    total_run = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_run - failures - errors - skipped
    success_rate = (passed / total_run * 100) if total_run > 0 else 0
    
    print("✅ Test execution completed!\n")
    
    print("="*80)
    print("📊 TEST RESULTS SUMMARY")
    print("="*80)
    
    # Main statistics box
    print(f"📈 Overall Results:")
    print(f"    🎯 Total Tests Run:     {total_run:>6}")
    print(f"    ✅ Passed:             {passed:>6} ({success_rate:5.1f}%)")
    print(f"    ❌ Failed:             {failures:>6}")
    print(f"    💥 Errors:             {errors:>6}")
    print(f"    ⏭️  Skipped:            {skipped:>6}")
    print(f"    ⏱️  Execution Time:     {execution_time:>6.2f}s")
    
    # Status indicator
    if failures == 0 and errors == 0:
        status = "🎉 ALL TESTS PASSED!"
        status_color = "✅"
    elif failures + errors < 5:
        status = "⚠️  MOSTLY SUCCESSFUL (Minor Issues)"
        status_color = "🟡"
    else:
        status = "❌ NEEDS ATTENTION (Multiple Failures)"
        status_color = "🔴"
    
    print(f"\n{status_color} Status: {status}")
    
    # Module breakdown
    if loaded_modules:
        print(f"\n📦 Module Breakdown:")
        for module_name, description, test_count in loaded_modules:
            print(f"    • {description}: {test_count} tests")
    
    # Show failures and errors if any
    if failures > 0:
        print(f"\n❌ FAILURES ({failures}):")
        for i, (test, traceback) in enumerate(result.failures, 1):
            test_name = str(test).split()[0]
            # Extract just the key error message
            error_lines = traceback.strip().split('\n')
            key_error = error_lines[-1] if error_lines else "Unknown error"
            print(f"    {i:2}. {test_name}")
            print(f"        💬 {key_error}")
    
    if errors > 0:
        print(f"\n💥 ERRORS ({errors}):")
        for i, (test, traceback) in enumerate(result.errors, 1):
            test_name = str(test).split()[0]
            # Extract just the key error message
            error_lines = traceback.strip().split('\n')
            key_error = error_lines[-1] if error_lines else "Unknown error"
            print(f"    {i:2}. {test_name}")
            print(f"        💬 {key_error}")
    
    if skipped > 0:
        print(f"\n⏭️  SKIPPED ({skipped}):")
        for i, (test, reason) in enumerate(result.skipped, 1):
            test_name = str(test).split()[0]
            print(f"    {i:2}. {test_name}")
            print(f"        💬 {reason}")
    
    # Performance insights
    if execution_time > 0:
        tests_per_second = total_run / execution_time
        print(f"\n⚡ Performance:")
        print(f"    • Tests per second: {tests_per_second:.1f}")
        print(f"    • Average test time: {execution_time/total_run*1000:.1f}ms")
    
    # Quick command suggestions
    print(f"\n🔧 Quick Commands:")
    print(f"    • Run specific module: python tests\\run_all_tests.py test_utilities")
    print(f"    • Run with discovery: python tests\\run_all_tests.py discover")
    print(f"    • Re-run failed tests: python -m pytest tests\\ --lf")
    
    print("="*80 + "\n")
    
    # Return success/failure status
    return failures == 0 and errors == 0

def run_specific_test(test_name):
    """Run a specific test module with clean output"""
    
    module_descriptions = {
        'test_models_comprehensive': 'Model Architecture & Components',
        'test_data_processing': 'Data Processing & Tokenization',
        'test_training_pipeline': 'Training Pipeline & Configuration', 
        'test_utilities': 'Utility Functions & Helpers',
        'test_integration': 'Integration & End-to-End Tests'
    }
    
    description = module_descriptions.get(test_name, test_name)
    
    print(f"\n🎯 Running Specific Test Module")
    print("="*60)
    print(f"📦 Module: {test_name}")
    print(f"📝 Description: {description}")
    print("="*60)
    
    try:
        start_time = time.time()
        module = importlib.import_module(f'tests.{test_name}')
        suite = unittest.TestLoader().loadTestsFromModule(module)
        test_count = suite.countTestCases()
        
        print(f"✅ Module loaded successfully ({test_count} tests)")
        print(f"🧪 Running tests...\n")
        
        # Use less verbose output for cleaner results
        stream = StringIO()
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=0,
            descriptions=False
        )
        result = runner.run(suite)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Calculate statistics
        total_run = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
        passed = total_run - failures - errors - skipped
        success_rate = (passed / total_run * 100) if total_run > 0 else 0
        
        print("📊 Test Results:")
        print(f"    🎯 Total Tests:    {total_run:>4}")
        print(f"    ✅ Passed:        {passed:>4} ({success_rate:5.1f}%)")
        print(f"    ❌ Failed:        {failures:>4}")
        print(f"    💥 Errors:        {errors:>4}")
        print(f"    ⏭️  Skipped:       {skipped:>4}")
        print(f"    ⏱️  Time:          {execution_time:>4.2f}s")
        
        if failures == 0 and errors == 0:
            print(f"\n🎉 All tests passed for {description}!")
        else:
            print(f"\n⚠️  Issues found in {description}")
            
            if failures > 0:
                print(f"\n❌ Failures:")
                for i, (test, traceback) in enumerate(result.failures, 1):
                    test_name = str(test).split()[0]
                    error_lines = traceback.strip().split('\n')
                    key_error = error_lines[-1] if error_lines else "Unknown error"
                    print(f"    {i}. {test_name}: {key_error}")
            
            if errors > 0:
                print(f"\n💥 Errors:")
                for i, (test, traceback) in enumerate(result.errors, 1):
                    test_name = str(test).split()[0]
                    error_lines = traceback.strip().split('\n')
                    key_error = error_lines[-1] if error_lines else "Unknown error"
                    print(f"    {i}. {test_name}: {key_error}")
        
        print("="*60 + "\n")
        
        return failures == 0 and errors == 0
        
    except ImportError as e:
        print(f"❌ Failed to import test module {test_name}: {e}")
        return False
    except Exception as e:
        print(f"💥 Error running {test_name}: {e}")
        return False

def run_test_discovery():
    """Run test discovery with clean output"""
    
    print("\n🔍 Test Discovery Mode")
    print("="*60)
    print("📦 Discovering tests in tests/ directory...")
    
    # Discover tests in the tests directory
    test_dir = Path(__file__).parent
    loader = unittest.TestLoader()
    
    try:
        start_time = time.time()
        
        suite = loader.discover(
            start_dir=str(test_dir),
            pattern='test_*.py',
            top_level_dir=str(project_root)
        )
        
        test_count = suite.countTestCases()
        print(f"✅ Discovered {test_count} tests")
        print(f"🧪 Running discovered tests...\n")
        
        # Use clean output
        stream = StringIO()
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=0,
            descriptions=False
        )
        result = runner.run(suite)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Calculate statistics
        total_run = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
        passed = total_run - failures - errors - skipped
        success_rate = (passed / total_run * 100) if total_run > 0 else 0
        
        print("📊 Discovery Results:")
        print(f"    🎯 Tests Found:    {test_count:>4}")
        print(f"    🏃 Tests Run:      {total_run:>4}")
        print(f"    ✅ Passed:        {passed:>4} ({success_rate:5.1f}%)")
        print(f"    ❌ Failed:        {failures:>4}")
        print(f"    💥 Errors:        {errors:>4}")
        print(f"    ⏭️  Skipped:       {skipped:>4}")
        print(f"    ⏱️  Time:          {execution_time:>4.2f}s")
        
        if failures == 0 and errors == 0:
            print(f"\n🎉 All discovered tests passed!")
        else:
            print(f"\n⚠️  Issues found in discovered tests")
        
        print("="*60 + "\n")
        
        return failures == 0 and errors == 0
        
    except Exception as e:
        print(f"❌ Test discovery failed: {e}")
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
