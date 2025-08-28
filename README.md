# arbeitsplanungstool

Installation Guide

You need Python 3.11.9
https://www.python.org/downloads/release/python-3119/

### Installer:

For HiWis: Use Admin priviliges: Untick
Add Python.exe to PATH: Tick!

### Now in VS Code:
Bottom right corner: Select Interpreter -> 
Python 3.11.9

Change path to repo if not automatically changed:
...\arbeitsplanungstool>

### Terminal:
py -3 -m venv .venv         # This creates a virtual environment (venv)
.\.venv\Scripts\Activate    # This activates the venv
# You should see something like (.venv) C:\Users\...\arbeitsplanungstool>
py -m pip install -r requirements.txt


### You're ready to go


### Troubleshooting
If PyQt6.Widgets is not working: pip install PyQt6 PyQt6-Qt6 PyQt6-sip
