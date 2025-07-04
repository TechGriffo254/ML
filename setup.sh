#!/bin/bash
echo "Setting up Python ML Environment..."
echo

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv ml_env
if [ $? -ne 0 ]; then
    echo "Error creating virtual environment"
    exit 1
fi

# Activate virtual environment and install packages
echo
echo "Activating virtual environment and installing packages..."
source ml_env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error installing packages"
    exit 1
fi

echo
echo "✅ Setup complete!"
echo
echo "To activate the environment in the future, run:"
echo "  source ml_env/bin/activate"
echo
echo "To start Jupyter Lab, run:"
echo "  jupyter lab"
echo
