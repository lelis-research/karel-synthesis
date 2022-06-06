# Karel Synthesis

This work-in-progress repository contains a symbolic implementation of the Karel the Robot DSL for symbolic program synthesis. It also contains a simple Bottom Up Search implementation for evaluation of the language.

The root folder contains sample scripts of the project usage:
- main_random.py: Generates a random program and executes it in pre-defined worlds.
- main_symbolic.py: Executes a program defined with the symbolic implementation.
- main_parsing.py: Executes a program defined from a string using the Parser class.
- main_bus.py: Executes Bottom Up Search in a single line of the Karel dataset.

## The Karel dataset

The script main_bus.py uses the sythetically generated [Karel dataset](https://msr-redmond.github.io/karel-dataset/) for the search. Download and extract it to /data.
