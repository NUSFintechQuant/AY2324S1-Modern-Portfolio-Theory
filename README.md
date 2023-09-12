# AY2324S1-Modern-Portfolio-Theory

Do check out our [contributing guidelines](CONTRIBUTING.md) for setting up the repository on your local machine and making pull requests to the github repository.

## Setup

We'll be using python 3.10 with some pip dependencies for now. We might switch to anaconda later but we'll stick to only esssential dependencies for now.

1. `python3.10 -m venv venv`
2. `source venv/bin/activate if Unix; if Windows: venv\Scripts\activate.bat`

## Risk Library

Tasks to do:
- [ ] statistical dispersion
- [ ] performance measures
- [ ] shortfall measures
- [ ] data scraping
    - just get a quick and dirty one that works for now. we'll probably refactor into something nicer later on anyway.

Some guidelines:
1. No need to worry too much about *where* the functions are located. Though we'll be putting them under the `risk` package for now.
1. Write *atomic* functions, i.e. the functions should not depend on or require external state.
1. Document your functions well; helps with readability and for other devs/researchers/yourself a few weeks from now.
1. Use static typing as much as possible.

## MPT
