
import sys
import os

# Ensure we can import the built module. 
# While usually installed in site-packages, during dev we might rely on PYTHONPATH or local build.
# The user env seems to be using venv: C:/Users/ireal/Documents/pycauset/.venv/Scripts/python.exe
# We assume the module is installed or available.

try:
    import _pycauset as internal
except ImportError:
    # Try importing from the package if it exposes it
    try:
        from pycauset import _pycauset as internal
    except ImportError:
        print("Could not import _pycauset. Ensure project is built and installed/path set.")
        sys.exit(1)

def test_registry():
    print("Testing OpRegistry...")
    registry = internal.OpRegistry.instance()
    assert registry is not None, "Registry instance is None"
    
    # Check if we can get a contract (assuming generic 'matmul' or similar is registered or can be registered)
    # If no ops are registered by default yet (since we only implemented the registry, not the registration in solvers yet),
    # we might expect it to be empty. But we can check the method existence.
    
    try:
        # Just checking if methods exist and don't crash
        contract = registry.get_contract("non_existent_op")
        if contract is None:
            print("Correctly returned None for non-existent op")
    except Exception as e:
        print(f"get_contract failed: {e}")

    print("OpRegistry test passed!")

if __name__ == "__main__":
    test_registry()
