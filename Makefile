.DEFAULT_GOAL := help

SHELL=/bin/bash

UNAME=$(shell uname -s)

.PHONY: install
install:  ## Install a virtual environment
	@poetry install -vv

.PHONY: fmt
fmt:  ## Run autoformatting and linting
	@poetry run pip install pre-commit
	@poetry run pre-commit install
	@poetry run pre-commit run --all-files

.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@git clean -X -d -f


.PHONY: help
help:  ## Display this help screen
	@echo
	@echo -e "\033[1mAvailable commands:\033[0m"
	@echo
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort
	@echo

.PHONY: test
test: install ## Run tests
	@poetry run pytest
