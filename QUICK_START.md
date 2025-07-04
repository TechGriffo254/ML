# Quick Start Guide

Welcome to your Python Machine Learning environment! Follow these steps to get started:

## 🚀 Quick Setup (Windows)

1. **Run the setup script:**
   ```cmd
   setup.bat
   ```

2. **Activate the environment:**
   ```cmd
   ml_env\Scripts\activate
   ```

3. **Start Jupyter Lab:**
   ```cmd
   jupyter lab
   ```

4. **Open the starter notebook:**
   Navigate to `notebooks/ml_starter_notebook.ipynb`

## 🐧 Quick Setup (Linux/Mac)

1. **Make setup script executable and run:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Activate the environment:**
   ```bash
   source ml_env/bin/activate
   ```

3. **Start Jupyter Lab:**
   ```bash
   jupyter lab
   ```

## 📚 What's Included

### Extensions Installed:
- ✅ Python
- ✅ Pylance  
- ✅ Jupyter
- ✅ Black Formatter
- ✅ autopep8

### Key Libraries:
- 📊 **Data Science**: pandas, numpy, scipy
- 📈 **Visualization**: matplotlib, seaborn, plotly
- 🤖 **Machine Learning**: scikit-learn, xgboost, lightgbm
- 📓 **Notebooks**: jupyter, ipykernel
- 🧪 **Testing**: pytest
- 🔧 **Development**: black, flake8

### Project Structure:
- 📁 `data/` - Store your datasets
- 📓 `notebooks/` - Jupyter notebooks for exploration
- 🐍 `src/` - Reusable Python modules
- 🤖 `models/` - Saved ML models
- 🧪 `tests/` - Unit tests

## 🎯 Next Steps

1. **Explore the starter notebook**: `notebooks/ml_starter_notebook.ipynb`
2. **Try the custom utilities**: Import from `src/` modules
3. **Run tests**: `pytest tests/`
4. **Add your data**: Place datasets in `data/raw/`

## 🆘 Troubleshooting

**Python not found?**
- Make sure Python 3.8+ is installed and in your PATH

**Package installation fails?**
- Try upgrading pip: `python -m pip install --upgrade pip`
- Check your internet connection

**VS Code not recognizing the environment?**
- Press `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose `ml_env/Scripts/python.exe`

**Happy coding! 🎉**
