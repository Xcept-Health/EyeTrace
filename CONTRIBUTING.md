# Contributing to EyeTrace

First off, thank you for considering contributing to EyeTrace! We welcome all forms of contributions, including bug reports, feature suggestions, documentation improvements, code, tests, and more.

This guide will help you understand how to contribute effectively.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Submitting a Pull Request](#submitting-a-pull-request)
- [Setting Up the Development Environment](#setting-up-the-development-environment)
- [Coding Conventions](#coding-conventions)
- [Testing Your Changes](#testing-your-changes)
- [Documentation](#documentation)
- [Review Process](#review-process)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a respectful and inclusive environment. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please open an issue on GitHub and follow these steps:

1. **Search** existing issues to see if the bug has already been reported.
2. Use a **clear and descriptive title**.
3. Provide **step-by-step instructions** to reproduce the problem.
4. Describe your **environment** (OS, Python version, dependency versions).
5. If possible, include **code snippets, screenshots, or log files**.

### Suggesting Enhancements

We welcome suggestions for new features or improvements! Please open an issue with:

- A **descriptive title**.
- A **detailed explanation** of the proposed functionality.
- **Use cases** and why it would be valuable.
- If relevant, **references** to existing work (papers, other implementations).

### Submitting a Pull Request

We love pull requests (PRs)! Here’s how to proceed:

1. **Fork** the repository and create a branch from `main` with a descriptive name (e.g., `feature/add-perclos`, `fix/correct-ear-calculation`).
2. **Implement** your changes following the coding conventions (see below).
3. **Test** your changes thoroughly (add tests if needed).
4. **Document** your changes (update docstrings, the documentation if applicable).
5. **Push** your branch to your fork and open a pull request against `main`.
6. In the PR description, clearly explain:
   - What changes you made
   - Why they are necessary
   - How to test them (if not obvious)
7. Link the PR to any related issues (e.g., "Closes #123").

## Setting Up the Development Environment

1. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/eyetrace.git
   cd eyetrace