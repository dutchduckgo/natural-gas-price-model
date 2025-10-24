# Contributing to Natural Gas Price Model

Thank you for your interest in contributing to the Natural Gas Price Model project! We welcome contributions from the community.

## 🤝 How to Contribute

### 1. Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/natural-gas-price-model.git
   cd natural-gas-price-model
   ```

### 2. Set Up Development Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install black flake8 mypy pytest

# Run tests to ensure everything works
python -m pytest tests/
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 4. Make Your Changes

- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 5. Code Style

We use the following tools for code quality:

```bash
# Format code
black src/ tests/

# Check for issues
flake8 src/ tests/

# Type checking
mypy src/
```

### 6. Commit and Push

```bash
git add .
git commit -m "Add your feature description"
git push origin feature/your-feature-name
```

### 7. Create a Pull Request

1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Fill out the PR template
4. Submit the PR

## 🎯 Areas for Contribution

### High Priority
- **Data Sources**: Implement additional data sources (CME, FERC, etc.)
- **Models**: Add new ML models (Prophet, ARIMA, etc.)
- **Features**: Implement additional feature engineering
- **Documentation**: Improve API documentation

### Medium Priority
- **Testing**: Add more comprehensive tests
- **Performance**: Optimize data processing
- **Visualization**: Add more plotting capabilities
- **Examples**: Create additional example notebooks

### Low Priority
- **UI**: Web interface for model training
- **Deployment**: Docker containerization
- **Monitoring**: Model performance monitoring

## 📝 Pull Request Guidelines

### Before Submitting
- [ ] Code follows the project's style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] No merge conflicts

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
```

## 🐛 Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: How to reproduce the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: OS, Python version, etc.
6. **Screenshots**: If applicable

## 💡 Feature Requests

When requesting features, please include:

1. **Use Case**: Why this feature would be useful
2. **Proposed Solution**: How you think it should work
3. **Alternatives**: Other solutions you've considered
4. **Additional Context**: Any other relevant information

## 📚 Development Setup

### Prerequisites
- Python 3.8+
- Git
- Virtual environment (recommended)

### Setup Steps
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/natural-gas-price-model.git
cd natural-gas-price-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e .

# Run tests
python -m pytest
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_models.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Writing Tests
- Place tests in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies

## 📖 Documentation

### Code Documentation
- Use docstrings for all functions and classes
- Follow Google docstring format
- Include type hints where possible

### Example
```python
def calculate_degree_days(self, temp_data: pd.DataFrame, base_temp: float = 65.0) -> pd.DataFrame:
    """
    Calculate heating and cooling degree days from temperature data.
    
    Args:
        temp_data: DataFrame with 'date' and 'temperature' columns
        base_temp: Base temperature for degree day calculation
        
    Returns:
        DataFrame with HDD and CDD columns
    """
    # Implementation here
```

## 🏷️ Release Process

1. Update version in `config.py`
2. Update `CHANGELOG.md`
3. Create release tag
4. Update documentation

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: For security issues or private matters

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to the Natural Gas Price Model project!
