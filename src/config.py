
import os
import sys

# Try to use tomllib (Python 3.11+) or fall back to tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("Error: 'tomli' or 'tomllib' is required. Please install dependencies.")
        sys.exit(1)

def load_config(config_path="settings.toml"):
    """
    Loads the configuration from a TOML file.
    Searches for settings.toml, then settings.example.toml.
    """
    # Environment flags for CI or Testing
    is_ci = os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CI") == "true"
    is_test = "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ
    
    # Filenames to try in order
    filenames = [config_path, "settings.example.toml"]
    
    # Directories to search in order
    search_dirs = ["."]
    
    # If running from src/ or similar, check parent
    search_dirs.append("..")
    search_dirs.append(os.path.join(os.path.dirname(__file__), ".."))
    
    # Support for PyInstaller (frozen application)
    if getattr(sys, 'frozen', False):
        search_dirs.insert(0, os.path.dirname(sys.executable))
    
    config = None
    loaded_path = None
    
    for filename in filenames:
        for directory in search_dirs:
            path = os.path.join(directory, filename)
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        config = tomllib.load(f)
                    loaded_path = path
                    break
                except Exception as e:
                    print(f"Failed to parse {path}: {e}")
                    sys.exit(1)
        if config:
            break
                
    if config is None:
        if is_ci or is_test:
            print("Warning: No configuration file found. Using empty fallback for CI/Testing.")
            return {}
        
        print(f"Error: Could not find configuration file '{config_path}' or 'settings.example.toml' in search paths.")
        print(f"Search directories: {[os.path.abspath(d) for d in search_dirs]}")
        sys.exit(1)
        
    if "example" in loaded_path:
        print(f"Note: Using template configuration from {os.path.abspath(loaded_path)}")
        
    return config

# Load config once when module is imported
# Users of this module can import 'settings' directly
settings = load_config()
