# Variables
FLASK_APP=app.py
VENV=venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

# Targets
.PHONY: all setup run clean

all: setup run

setup: $(VENV)

$(VENV):
    python3 -m venv $(VENV)
    $(PIP) install -r requirements.txt

run:
    FLASK_APP=$(FLASK_APP) $(PYTHON) -m flask run

clean:
    rm -rf $(VENV) __pycache__ *.pyc *.pyo *.pyd