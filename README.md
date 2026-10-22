# Final Report and Project Software

## How to Run the Code
The below assumes that the user has `python` and `pip` already installed and accessible from the project directory.
Additionally, the commands below are for Windows users - Mac or Linux users should run the equivalent commands.

The easiest way to run the code is to first create a virtual environment using
```
python -m venv .venv
```
Then follow the steps below.

### Dependencies
Dependencies are listed in `requirements.txt`, and can be easily downloaded all at once using
```
pip install -r requirements.txt
```
The project requires `numba`, `numpy`, `matplotlib`, `ipykernel`, and `scipy`.

### Code Structure
Source code is structured as a Python module in `project/sandpiles`. 
To reproduce the experiment, the Jupyter notebook `project/sandpiles.ipynb` contains structured glue code and code to generate all figures included in the report. It imports and calls functions defined in the `project/sandpiles` module.

Final Report template



TEST EDIT 



The PDF for the report should be generated from main.tex:



Note for Latex or Overleaf users: (Credit to Antonio Spataro)
You may need to move a line between the preamble.tex file to the beginning of main.tex (for Overleaf) or vice versa:
\\documentclass\[a4paper, 10.5pt, oneside]{report}

