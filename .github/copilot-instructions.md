<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Python Machine Learning Project Instructions

This is a Python machine learning project focused on data science and ML development. Please follow these guidelines when generating code:

## Code Style
- Use Python type hints wherever possible
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Use black formatter for code formatting

## Project Structure
- Place reusable functions in the `src/` directory
- Use notebooks for exploration and prototyping
- Store raw data in `data/raw/` and processed data in `data/processed/`
- Save trained models in the `models/` directory

## Libraries and Frameworks
- Prefer pandas for data manipulation
- Use numpy for numerical operations
- Use matplotlib/seaborn for basic plotting, plotly for interactive visualizations
- Use scikit-learn for traditional ML algorithms
- Include proper error handling and validation

## Best Practices
- Always include data validation and error handling
- Use descriptive commit messages
- Write unit tests for critical functions
- Document complex algorithms and data transformations
- Use logging instead of print statements for debugging
