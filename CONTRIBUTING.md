# Contributing to ComfyUI-LLM-Session

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs
- Check existing issues first
- Provide detailed reproduction steps
- Include ComfyUI version, Python version, and OS
- Share relevant log outputs

### Suggesting Features
- Open an issue with the "enhancement" label
- Describe the use case and expected behavior
- Explain why this feature would be useful

### Submitting Code
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to your fork (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Code Guidelines

### Python Style
- Follow PEP 8
- Use meaningful variable names
- Add docstrings for functions
- Keep functions focused and concise

### ComfyUI Node Guidelines
- Use consistent INPUT_TYPES structure
- Provide clear tooltips
- Handle errors gracefully with user-friendly messages
- Test with multiple model types

### Licensing
- All contributions must be GPL-3.0 compatible
- Add copyright header to new files
- Respect third-party licenses

## Development Setup

```bash
git clone https://github.com/kantan-kanto/ComfyUI-LLM-Session.git
cd ComfyUI-LLM-Session
pip install -r requirements.txt
```

## Testing
- Test with both local and API models
- Verify error handling
- Check compatibility with latest ComfyUI

## Questions?
Open an issue for questions or discussions.
