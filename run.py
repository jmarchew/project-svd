import sys
import os

# Add src/ to python path
BASE_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

# Import and run your SVD module
import svd.mysvd