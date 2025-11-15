---
applyTo: "**/*.py"
---
# General Guidelines
- I'm not always right, so be critical of my suggestions.
- Ask me to explain if you don't understand a suggestion.
# Project coding standards for Python
- Activate conda environment before running or editing code. It's name is 'venv_ovitraps'.
- Follow the PEP 8 style guide for Python.
- Always prioritize readability and clarity.
- Write clear and concise comments for each function.
- Ensure functions have descriptive names and include type hints.
- Maintain proper indentation.
- Always test code after writing it but avoid changing data files and databases. Prefer using pytest and dvc for testing.
- Avoid hardcoding values; use configuration files or constants instead.
- Don't ask me to run terminal commands. Just do it yourself. If necessary, I'll allow your commands. 
# Test Instructions
- Use pytest for running tests. Avoid unittest as much as possible.
- Avoid randomness in tests. Use fixed values or set random seeds, unless testing randomness itself.
- Place test files in the 'tests' directory, with a structure that mirrors the root directory.
- Use fixtures for setting up test data and environments.
- Test separate functional units independently. This includes functions and class methods.
- Write tests that cover edge cases and typical use cases using parametrization.
- Write tests that cover invalid inputs and error handling.
- Mock external dependencies to isolate the code under test if needed. When doing so, ensure that the mocks were called the exact number of times expected with the correct arguments.
- Use shared_tests_utils.py functions to evaluate common features. All tested function must include: test_function_signature_and_return, test_docstring_exists, test_performance_timing, test_function_determinism. Functions that receives files as inputs must use "test_file_integrity". Functions that process dataframes or numpy arrays must use "test_data_integrity". You can modify the functions in shared_tests_utils.py if needed.