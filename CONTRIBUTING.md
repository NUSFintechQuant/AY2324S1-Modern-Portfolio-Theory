# Contributing

Whether you're new to Git or a seasoned user, do give this a read. This document provides guidelines and steps for contributing.

## Getting Started with Git

1. **Clone the Repository**: Start by cloning the main repository to your local machine.
```bash
git clone https://github.com/NUSFintechQuant/AY2324S1-Modern-Portfolio-Theory.git
```

2. **Navigate to the Directory**:
```bash
cd AY2324S1-Modern-Portfolio-Theory
```

3. **IMPORTANT: Stay Updated**: Before starting any new work, always ensure you have the latest changes from the main branch:
```bash
git pull origin main
```

## Making Contributions

1. **Create a New Branch**: Before making any changes, always create a new branch for your work. This helps isolate your topic and makes it easier to integrate changes:
```bash
git checkout -b feature-name-or-bugfix
```

2. **Make Your Changes**: Work on your feature or bugfix in this branch.

3. **Commit Your Work**: Once you've made your changes, commit them:
```bash
git add .
git commit -m "Descriptive message about the changes you've made"
```

4. **Push Your Branch**: Push your branch with the new changes to the GitHub repository:
```bash
git push origin feature-name-or-bugfix
```

5. **Open a Pull Request (PR)**: Return to the GitHub page of the main repository and click on "Pull request". Start a new pull request from your recent branch to the main branch. Fill in the necessary details and create the PR. Do tag your PR if possible:
1. `feat`: Introducing a new feature.
1. `fix`: A bug fix.
1. `docs`: Documentation only changes.
1. `refactor`: A code change that neither fixes a bug nor adds a feature.
1. `test`: Adding missing tests or correcting existing tests.
1. `chore`: Changes to the build process or auxiliary tools and libraries such as documentation generation.

## Guidelines

1. **Documentation**: Ensure your functions are well-documented for readability and clarity.
1. **Use Static Typing**: Incorporate static typing as much as possible.

## After Making a Pull Request (PR)

Once your PR is open, the relevant stakeholders will review your changes. They will be minrei and gavin for now, but it might be you as you carve your stake in the project. You might be asked to make modifications before your changes are merged.

Thank you for your contribution!
