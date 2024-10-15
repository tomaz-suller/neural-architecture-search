#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = nas
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	pre-commit install




## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff check nas

## Fix ruff warnings
.PHONY: fix
fix:
	ruff fix nas

## Format source code with ruff
.PHONY: format
format:
	ruff format nas




## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	@$(PYTHON_INTERPRETER) -m venv venv
	@echo ">>> New venv created. Activate with:\nsource venv/bin/activate"




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

REMOTE_SERVER = nas
REMOTE_REPO_ROOT = ~/neural-architecture-search
REMOTE = ${REMOTE_SERVER}:${REMOTE_REPO_ROOT}
RSYNC = rsync \
		--archive \
		--human-readable \
		--partial \
		--recursive \
		--update \
		--info=PROGRESS2

## Spin up MLFlow server
mlflow:
	mlflow server --backend-store-uri results/mlruns/

## Upload experiment files to remote server
upload:
	${RSYNC} --filter=':- .gitignore' . ${REMOTE}

## Download results from remote server
download:
	${RSYNC} ${REMOTE}/results .

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
