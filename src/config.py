
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
    Helper to resolve path relative to project root if needed.
    """
    # If running from src/, we might need to go up one level or check current dir
    # We will look for settings.toml in the current working directory or parent directory
    
    search_paths = [
        config_path,
        os.path.join("..", config_path),
        os.path.join(os.path.dirname(__file__), "..", config_path)
    ]
    
    # Support for PyInstaller (frozen application)
    if getattr(sys, 'frozen', False):
        # When frozen, looks for config next to the executable
        # sys.executable points to the .exe file
        exe_dir = os.path.dirname(sys.executable)
        search_paths.insert(0, os.path.join(exe_dir, config_path))
    
    config = None
    loaded_path = None
    
    for path in search_paths:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    config = tomllib.load(f)
                loaded_path = path
                break
            except Exception as e:
                print(f"Failed to parse {path}: {e}")
                sys.exit(1)
                
    if config is None:
        print(f"Error: Could not find configuration file '{config_path}' in search paths.")
        print(f"Search paths: {[os.path.abspath(p) for p in search_paths]}")
        sys.exit(1)
        
    return config

# Load config once when module is imported
# Users of this module can import 'settings' directly
settings = load_config()
