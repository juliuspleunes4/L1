#!/usr/bin/env python3
"""
@file       : view_latest_results.py
@author     : J.J.G. Pleunes
@date       : 08/2025
@brief      : Utility to view latest test results
@details    : Opens the most recent test report in browser or displays summary
@version    : 1.0

@license    : MIT License
Copyright (c) 2025 Julius Pleunes
"""

import os
import sys
import json
import webbrowser
from pathlib import Path
from datetime import datetime

def find_latest_results():
    """Find the most recent test results"""
    
    test_dir = Path(__file__).parent
    results_dir = test_dir / "results"
    logs_dir = test_dir / "logs"
    
    if not results_dir.exists():
        return None, "No results directory found. Run tests first."
    
    # Find latest HTML report
    html_files = list(results_dir.glob("test_report_*.html"))
    if not html_files:
        return None, "No test reports found. Run tests first."
    
    # Sort by modification time and get latest
    latest_html = max(html_files, key=lambda f: f.stat().st_mtime)
    
    # Find corresponding JSON file
    timestamp = latest_html.stem.replace("test_report_", "")
    json_file = results_dir / f"test_results_{timestamp}.json"
    csv_file = results_dir / f"test_results_{timestamp}.csv"
    log_file = logs_dir / f"test_run_{timestamp}.log"
    
    return {
        'html': latest_html,
        'json': json_file if json_file.exists() else None,
        'csv': csv_file if csv_file.exists() else None,
        'log': log_file if log_file.exists() else None,
        'timestamp': timestamp
    }, None

def display_summary(json_file):
    """Display a quick summary from JSON results"""
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        stats = data['statistics']
        timestamp = datetime.fromisoformat(data['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n" + "="*60)
        print("📊 LATEST TEST RESULTS SUMMARY")
        print("="*60)
        print(f"🕒 Generated: {timestamp}")
        print(f"⏱️  Execution Time: {data['execution_time']:.2f}s")
        print()
        print(f"📈 Results:")
        print(f"    🎯 Total Tests:    {stats['total_tests']:>6}")
        print(f"    ✅ Passed:        {stats['passed']:>6} ({stats['success_rate']:5.1f}%)")
        print(f"    ❌ Failed:        {stats['failed']:>6}")
        print(f"    💥 Errors:        {stats['errors']:>6}")
        print(f"    ⏭️  Skipped:       {stats['skipped']:>6}")
        
        if stats['failed'] == 0 and stats['errors'] == 0:
            print(f"\n🎉 Status: ALL TESTS PASSED!")
        elif stats['failed'] + stats['errors'] < 5:
            print(f"\n🟡 Status: MOSTLY SUCCESSFUL (Minor Issues)")
        else:
            print(f"\n🔴 Status: NEEDS ATTENTION (Multiple Failures)")
        
        print("\n📦 Modules:")
        for module in data['modules']['loaded_modules']:
            print(f"    ✅ {module['description']}: {module['test_count']} tests")
        
        for module in data['modules']['failed_modules']:
            print(f"    ❌ {module['description']}: {module['error'][:50]}...")
        
        if data['failures']:
            print(f"\n❌ Key Failures:")
            for i, failure in enumerate(data['failures'][:3], 1):
                test_name = failure['test'].split('.')[-1]
                print(f"    {i}. {test_name}: {failure['error']}")
            if len(data['failures']) > 3:
                print(f"    ... and {len(data['failures']) - 3} more")
        
        if data['errors']:
            print(f"\n💥 Key Errors:")
            for i, error in enumerate(data['errors'][:3], 1):
                test_name = error['test'].split('.')[-1]
                print(f"    {i}. {test_name}: {error['error']}")
            if len(data['errors']) > 3:
                print(f"    ... and {len(data['errors']) - 3} more")
        
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error reading results: {e}")

def main():
    """Main function"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
    else:
        command = "summary"
    
    files, error = find_latest_results()
    
    if error:
        print(f"❌ {error}")
        print("\n💡 Run tests first: python tests\\run_all_tests.py")
        return 1
    
    if command in ["html", "browser", "open"]:
        # Open HTML report in browser
        try:
            webbrowser.open(f"file:///{files['html'].as_posix()}")
            print(f"🌐 Opening HTML report in browser...")
            print(f"📄 File: {files['html']}")
        except Exception as e:
            print(f"❌ Failed to open browser: {e}")
            print(f"📄 Manual path: {files['html']}")
        
    elif command in ["json", "data"]:
        # Display JSON file path or content
        if files['json']:
            if len(sys.argv) > 2 and sys.argv[2] == "content":
                try:
                    with open(files['json'], 'r') as f:
                        print(f.read())
                except Exception as e:
                    print(f"❌ Error reading JSON: {e}")
            else:
                print(f"📄 JSON file: {files['json']}")
                print(f"💡 View content: python tests\\view_latest_results.py json content")
        else:
            print("❌ No JSON file found")
    
    elif command in ["csv", "spreadsheet"]:
        # Display CSV file path
        if files['csv']:
            print(f"📊 CSV file: {files['csv']}")
            print(f"💡 Open with: start {files['csv']}")
        else:
            print("❌ No CSV file found")
    
    elif command in ["log", "logs"]:
        # Display log file path or content
        if files['log']:
            if len(sys.argv) > 2 and sys.argv[2] == "content":
                try:
                    with open(files['log'], 'r') as f:
                        print(f.read())
                except Exception as e:
                    print(f"❌ Error reading log: {e}")
            else:
                print(f"📝 Log file: {files['log']}")
                print(f"💡 View content: python tests\\view_latest_results.py log content")
        else:
            print("❌ No log file found")
    
    elif command in ["summary", "quick", "status"]:
        # Display quick summary
        if files['json']:
            display_summary(files['json'])
        else:
            print("❌ No results file found for summary")
    
    elif command in ["files", "list", "all"]:
        # List all available files
        print(f"\n📁 Latest Test Results ({files['timestamp']}):")
        print(f"    🌐 HTML Report: {files['html']}")
        if files['json']:
            print(f"    📄 JSON Data:   {files['json']}")
        if files['csv']:
            print(f"    📊 CSV Data:    {files['csv']}")
        if files['log']:
            print(f"    📝 Log File:    {files['log']}")
        
        print(f"\n🔧 Quick Commands:")
        print(f"    • View summary:     python tests\\view_latest_results.py")
        print(f"    • Open HTML:        python tests\\view_latest_results.py html")
        print(f"    • Show JSON:        python tests\\view_latest_results.py json")
        print(f"    • Show logs:        python tests\\view_latest_results.py log")
        print(f"    • List all files:   python tests\\view_latest_results.py files")
    
    else:
        print("🔧 Test Results Viewer")
        print("="*40)
        print("Usage: python tests\\view_latest_results.py [command]")
        print()
        print("Commands:")
        print("  summary, quick    - Show quick summary (default)")
        print("  html, browser     - Open HTML report in browser")
        print("  json, data        - Show JSON file path")
        print("  csv, spreadsheet  - Show CSV file path") 
        print("  log, logs         - Show log file path")
        print("  files, list       - List all result files")
        print()
        print("Examples:")
        print("  python tests\\view_latest_results.py")
        print("  python tests\\view_latest_results.py html")
        print("  python tests\\view_latest_results.py json content")
        print("  python tests\\view_latest_results.py log content")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
