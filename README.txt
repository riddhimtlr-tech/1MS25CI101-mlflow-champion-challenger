STUDENT PYTHON SETUP

WINDOWS
1. Install Python 3 if it is not already installed:
   https://www.python.org/downloads/windows/
   During installation, enable "Add python.exe to PATH" if shown.
2. Keep setup.bat and requirements.txt in the same project folder.
3. Double-click setup.bat and wait for SETUP COMPLETE.
4. In PyCharm, open Settings > Project > Python Interpreter.
5. Add/select an existing interpreter and choose:
   <project folder>\.venv\Scripts\python.exe

macOS / LINUX
1. Ensure Python 3 is installed.
2. Open Terminal in the project folder.
3. Run: chmod +x setup.sh
4. Run: ./setup.sh
5. In PyCharm, select:
   <project folder>/.venv/bin/python

NOTES
- The scripts create a local .venv folder and install packages from requirements.txt.
- An internet connection is required for package installation.
- Keep setup.bat/setup.sh and requirements.txt in the project folder.
- Do not upload the .venv folder to GitHub or copy it between different operating systems.
