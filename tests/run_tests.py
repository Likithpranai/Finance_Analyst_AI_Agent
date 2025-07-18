#!/usr/bin/env python3
"""
Test runner for the Finance Analyst AI Agent.
This script discovers and runs all tests in the tests directory.
"""

import unittest
import sys
import os

def run_all_tests():
    """Discover and run all tests in the tests directory."""
    # Add the parent directory to the path so we can import modules
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(__file__), pattern='test_*.py')
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return non-zero exit code if tests failed
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_all_tests())
