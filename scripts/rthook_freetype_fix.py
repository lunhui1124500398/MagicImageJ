import os
import sys
import ctypes

def _load_freetype_dll():
    """
    Runtime hook to explicitly load libfreetype.dll from the frozen bundle.
    This helps when standard ctypes.util.find_library fails or when
    Windows fails to look in the correct directory.
    """
    if not getattr(sys, 'frozen', False):
        return

    base_path = sys._MEIPASS
    
    # Common locations where PyInstaller might put the DLL
    potential_paths = [
        os.path.join(base_path, 'freetype', 'libfreetype.dll'),
        os.path.join(base_path, 'libfreetype.dll'),
        os.path.join(base_path, 'pkgs', 'freetype', 'libfreetype.dll'),
    ]

    dll_loaded = False
    for dll_path in potential_paths:
        if os.path.exists(dll_path):
            try:
                # Load with absolute path to ensure dependencies are resolved if possible
                # RTLD_GLOBAL is not available on Windows, but standard CDLL helps
                ctypes.CDLL(dll_path)
                
                # Also AddDllDirectory if available (Python 3.8+ on Windows)
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(os.path.dirname(dll_path))
                
                # Set environment variable as fallback for some libs
                os.environ['FREETYPE_LIBRARY'] = dll_path
                
                dll_loaded = True
                print(f"Succefully pre-loaded freetype DLL from: {dll_path}")
                break
            except Exception as e:
                print(f"Failed to load freetype DLL from {dll_path}: {e}")

    if not dll_loaded:
        print("WARNING: Could not find or load libfreetype.dll in standard locations.")

# Execute the loader
_load_freetype_dll()
