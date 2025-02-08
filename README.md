# Frozen lake Game

## Project dependencies
- Step 1: Update Package List
    - sudo apt update
- Step 2: Install Python and Pip
    - sudo apt install python3 python3-pip
- Step 3: Install the Libraries
    - pip3 install gymnasium numpy matplotlib
- Step 4: Verify Installation
    - python3 -c "import gymnasium; import numpy; import matplotlib; print('Libraries installed successfully\!')"
- gymnasium.error.DependencyNotInstalled: pygame is not installed, run 
    - pip install "gymnasium[toy-text]"
