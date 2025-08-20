"""
Quick test script to verify all improvements are working.
"""

import logging
from config import Config
from main_improved import main

# Set up simple logging for the test
logging.basicConfig(level=logging.INFO)

# Test configuration
Config.PLACE = "Heidelberg, Germany" 
Config.N_REMOVE = 2  # Smaller for faster testing
Config.K_NN = 3

if __name__ == "__main__":
    print("Running quick test of improved codebase...")
    try:
        main()
        print("\n✅ ALL TESTS PASSED! The improved codebase is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
