include .env
export

setup:
	@test -f .env || cp .env-dist .env
	pip install -e .

# Package

update-pip:
	python -m pip install --upgrade pip

build-package:
	@if [ -d build ]; then rm -r build; fi
	@if [ -d dist ]; then rm -r dist; fi
	python setup.py bdist_wheel

