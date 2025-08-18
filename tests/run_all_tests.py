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
import json
import csv
import logging
from datetime import datetime
from io import StringIO
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_test_logging():
    """Setup logging directories and files"""
    test_dir = Path(__file__).parent
    logs_dir = test_dir / "logs"
    results_dir = test_dir / "results"
    
    # Create directories if they don't exist
    logs_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return {
        'logs_dir': logs_dir,
        'results_dir': results_dir,
        'timestamp': timestamp,
        'log_file': logs_dir / f"test_run_{timestamp}.log",
        'csv_file': results_dir / f"test_results_{timestamp}.csv",
        'json_file': results_dir / f"test_results_{timestamp}.json",
        'html_file': results_dir / f"test_report_{timestamp}.html"
    }

def save_test_results(result, execution_time, loaded_modules, failed_modules, files_info):
    """Save test results in multiple formats"""
    
    # Calculate statistics
    total_run = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_run - failures - errors - skipped
    success_rate = (passed / total_run * 100) if total_run > 0 else 0
    
    # Prepare test data
    test_data = {
        'timestamp': datetime.now().isoformat(),
        'execution_time': execution_time,
        'statistics': {
            'total_tests': total_run,
            'passed': passed,
            'failed': failures,
            'errors': errors,
            'skipped': skipped,
            'success_rate': success_rate
        },
        'modules': {
            'loaded': len(loaded_modules),
            'failed': len(failed_modules),
            'loaded_modules': [{'name': name, 'description': desc, 'test_count': count} 
                              for name, desc, count in loaded_modules],
            'failed_modules': [{'name': name, 'description': desc, 'error': error} 
                              for name, desc, error in failed_modules]
        },
        'failures': [{'test': str(test), 'error': traceback.strip().split('\n')[-1]} 
                    for test, traceback in result.failures],
        'errors': [{'test': str(test), 'error': traceback.strip().split('\n')[-1]} 
                  for test, traceback in result.errors],
        'skipped': [{'test': str(test), 'reason': reason} 
                   for test, reason in (result.skipped if hasattr(result, 'skipped') else [])]
    }
    
    # Save JSON results
    try:
        with open(files_info['json_file'], 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        print(f"📄 JSON report saved: {files_info['json_file']}")
    except Exception as e:
        print(f"⚠️  Failed to save JSON report: {e}")
    
    # Save CSV results
    try:
        with open(files_info['csv_file'], 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write summary
            writer.writerow(['Test Run Summary', files_info['timestamp']])
            writer.writerow(['Total Tests', total_run])
            writer.writerow(['Passed', passed])
            writer.writerow(['Failed', failures])
            writer.writerow(['Errors', errors])
            writer.writerow(['Skipped', skipped])
            writer.writerow(['Success Rate %', f"{success_rate:.1f}"])
            writer.writerow(['Execution Time (s)', f"{execution_time:.2f}"])
            writer.writerow([])
            
            # Write detailed results
            writer.writerow(['Test Name', 'Status', 'Details'])
            
            # Add failures
            for test, traceback in result.failures:
                error_msg = traceback.strip().split('\n')[-1]
                writer.writerow([str(test), 'FAILED', error_msg])
            
            # Add errors
            for test, traceback in result.errors:
                error_msg = traceback.strip().split('\n')[-1]
                writer.writerow([str(test), 'ERROR', error_msg])
            
            # Add skipped
            for test, reason in (result.skipped if hasattr(result, 'skipped') else []):
                writer.writerow([str(test), 'SKIPPED', reason])
        
        print(f"📊 CSV report saved: {files_info['csv_file']}")
    except Exception as e:
        print(f"⚠️  Failed to save CSV report: {e}")
    
    # Save HTML report
    try:
        html_content = generate_html_report(test_data, files_info)
        with open(files_info['html_file'], 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"🌐 HTML report saved: {files_info['html_file']}")
    except Exception as e:
        print(f"⚠️  Failed to save HTML report: {e}")
    
    # Setup and save detailed log
    try:
        logging.basicConfig(
            filename=files_info['log_file'],
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='w'
        )
        
        logger = logging.getLogger('test_runner')
        logger.info(f"L1 Test Suite Execution Report")
        logger.info(f"Execution Time: {execution_time:.2f} seconds")
        logger.info(f"Total Tests: {total_run}, Passed: {passed}, Failed: {failures}, Errors: {errors}, Skipped: {skipped}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        for name, desc, count in loaded_modules:
            logger.info(f"Loaded Module: {name} ({desc}) - {count} tests")
        
        for name, desc, error in failed_modules:
            logger.error(f"Failed Module: {name} ({desc}) - {error}")
        
        for test, traceback in result.failures:
            logger.error(f"FAILURE: {test}")
            logger.error(f"Details: {traceback.strip().split(chr(10))[-1]}")
        
        for test, traceback in result.errors:
            logger.error(f"ERROR: {test}")
            logger.error(f"Details: {traceback.strip().split(chr(10))[-1]}")
        
        print(f"📝 Detailed log saved: {files_info['log_file']}")
    except Exception as e:
        print(f"⚠️  Failed to save log file: {e}")

def generate_html_report(test_data, files_info):
    """Generate an HTML test report"""
    
    stats = test_data['statistics']
    timestamp = files_info['timestamp']
    
    # Determine status color
    if stats['failed'] == 0 and stats['errors'] == 0:
        status_class = "success"
        status_text = "ALL TESTS PASSED"
    elif stats['failed'] + stats['errors'] < 5:
        status_class = "warning"
        status_text = "MOSTLY SUCCESSFUL"
    else:
        status_class = "danger"
        status_text = "NEEDS ATTENTION"
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>L1 Test Report - {timestamp}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f8f9fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-number {{ font-size: 2em; font-weight: bold; }}
        .success {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .danger {{ color: #e74c3c; }}
        .muted {{ color: #7f8c8d; }}
        .status-badge {{ padding: 8px 16px; border-radius: 20px; font-weight: bold; color: white; }}
        .status-badge.success {{ background: #27ae60; }}
        .status-badge.warning {{ background: #f39c12; }}
        .status-badge.danger {{ background: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #34495e; color: white; }}
        .failed {{ background: #fdf2f2; }}
        .error {{ background: #fef5e7; }}
        .skipped {{ background: #f8f9fa; }}
        .module-list {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin: 20px 0; }}
        .module-card {{ border: 1px solid #ddd; border-radius: 6px; padding: 15px; }}
        .module-card.loaded {{ border-left: 4px solid #27ae60; }}
        .module-card.failed {{ border-left: 4px solid #e74c3c; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 L1 Project Test Report</h1>
        <p><strong>Generated:</strong> {test_data['timestamp']}</p>
        <p><strong>Execution Time:</strong> {test_data['execution_time']:.2f} seconds</p>
        <p><span class="status-badge {status_class}">{status_text}</span></p>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{stats['total_tests']}</div>
                <div>Total Tests</div>
            </div>
            <div class="stat-card">
                <div class="stat-number success">{stats['passed']}</div>
                <div>Passed ({stats['success_rate']:.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-number danger">{stats['failed']}</div>
                <div>Failed</div>
            </div>
            <div class="stat-card">
                <div class="stat-number warning">{stats['errors']}</div>
                <div>Errors</div>
            </div>
            <div class="stat-card">
                <div class="stat-number muted">{stats['skipped']}</div>
                <div>Skipped</div>
            </div>
        </div>
        
        <h2>📦 Test Modules</h2>
        <div class="module-list">'''
    
    # Add loaded modules
    for module in test_data['modules']['loaded_modules']:
        html += f'''
            <div class="module-card loaded">
                <h4>✅ {module['description']}</h4>
                <p><strong>Module:</strong> {module['name']}</p>
                <p><strong>Tests:</strong> {module['test_count']}</p>
            </div>'''
    
    # Add failed modules
    for module in test_data['modules']['failed_modules']:
        html += f'''
            <div class="module-card failed">
                <h4>❌ {module['description']}</h4>
                <p><strong>Module:</strong> {module['name']}</p>
                <p><strong>Error:</strong> {module['error'][:100]}...</p>
            </div>'''
    
    html += '</div>'
    
    # Add failures and errors tables
    if test_data['failures'] or test_data['errors']:
        html += '<h2>❌ Issues Found</h2>'
        
        if test_data['failures']:
            html += '<h3>Failures</h3><table><tr><th>Test</th><th>Error</th></tr>'
            for failure in test_data['failures']:
                html += f'<tr class="failed"><td>{failure["test"]}</td><td>{failure["error"]}</td></tr>'
            html += '</table>'
        
        if test_data['errors']:
            html += '<h3>Errors</h3><table><tr><th>Test</th><th>Error</th></tr>'
            for error in test_data['errors']:
                html += f'<tr class="error"><td>{error["test"]}</td><td>{error["error"]}</td></tr>'
            html += '</table>'
    
    if test_data['skipped']:
        html += '<h2>⏭️ Skipped Tests</h2><table><tr><th>Test</th><th>Reason</th></tr>'
        for skipped in test_data['skipped']:
            html += f'<tr class="skipped"><td>{skipped["test"]}</td><td>{skipped["reason"]}</td></tr>'
        html += '</table>'
    
    html += '''
        <hr style="margin: 30px 0;">
        <p style="text-align: center; color: #7f8c8d;">
            Generated by L1 Test Runner • 
            <a href="https://github.com/juliuspleunes4/L1">L1 Project</a>
        </p>
    </div>
</body>
</html>'''
    
    return html

def run_all_tests():
    """Run all test modules with comprehensive reporting"""
    
    # Setup logging
    files_info = setup_test_logging()
    
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
    print(f"📁 Results will be saved to: tests/logs/ and tests/results/")
    
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
    
    # Save results to files
    print(f"\n💾 Saving Results:")
    save_test_results(result, execution_time, loaded_modules, failed_modules, files_info)
    
    # Quick command suggestions
    print(f"\n🔧 Quick Commands:")
    print(f"    • Run specific module: python tests\\run_all_tests.py test_utilities")
    print(f"    • Run with discovery: python tests\\run_all_tests.py discover")
    print(f"    • View HTML report: start {files_info['html_file']}")
    print(f"    • View latest results: type {files_info['json_file']}")
    
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
