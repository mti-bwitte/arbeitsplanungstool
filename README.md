# arbeitsplanungstool

Installation Guide

You need Python 3.11.9
https://www.python.org/downloads/release/python-3119/

# Installer:

For HiWis: Use Admin priviliges: Untick
Add Python.exe to PATH: Tick!

# Now in VS Code:
Install Python extension in VSCode Extensions

Bottom right corner: Select Interpreter -> Python 3.11.9

Change path to repo if not automatically changed:
...\arbeitsplanungstool>

# Terminal:
py -3 -m venv .venv         # This creates a virtual environment (venv)
Windows:
.\.venv\Scripts\Activate    # This activates the venv
Mac: 
source venv/bin/activate    # This activates the venv

### You should now see something like (.venv) C:\Users\...\arbeitsplanungstool>

py -m pip install -r requirements.txt  # Installing all required packages

# You're ready to go


### Troubleshooting
If PyQt6.Widgets is not working: pip install PyQt6 PyQt6-Qt6 PyQt6-sip
If any module is not found: pip install 'module name'
