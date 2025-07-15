<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# L1 LLM Project - GitHub Copilot Instructions

## Project Overview
This is a large language model (LLM) project called L1, built using Python and PyTorch. The project implements a transformer-based architecture for training and deploying language models.

## Code Style and Standards
- Follow PEP 8 Python style guidelines
- Use type hints for all function parameters and return values
- Write comprehensive docstrings for all classes and functions using Google style
- Keep functions focused and single-purpose
- Use descriptive variable and function names

## Architecture Guidelines
- Model components should be modular and reusable
- Use PyTorch conventions for model architecture
- Implement proper error handling and logging
- Follow the repository structure for file organization
- Use configuration files for all hyperparameters

## AI/ML Best Practices
- Always include proper tensor dimension comments
- Use appropriate PyTorch modules (nn.Module, nn.Parameter, etc.)
- Implement gradient checkpointing for memory efficiency
- Include proper device handling (CPU/GPU)
- Add assertions for tensor shapes where appropriate

## Testing Requirements
- Write unit tests for all core functionality
- Include integration tests for training pipeline
- Test model serialization/deserialization
- Verify gradient computation correctness
- Test data loading and preprocessing

## Documentation Standards
- Document all model parameters and their effects
- Include usage examples in docstrings
- Explain mathematical concepts and formulas
- Provide clear setup and installation instructions
- Document API endpoints and request/response formats

## Dependencies and Imports
- Use relative imports within the project
- Group imports: standard library, third-party, local imports
- Avoid circular imports
- Use lazy imports for heavy dependencies when possible
