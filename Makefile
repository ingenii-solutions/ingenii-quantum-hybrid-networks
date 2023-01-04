setup:
	@test -f .pypirc || cp .pypirc-dist .pypirc
	pip install -e .

update-pip:
	python -m pip install --upgrade pip

# Package

clean:
	@make clean-lint
	@make clean-package

clean-package:
	@rm -rf ./build ./dist ./*.egg-info ./.eggs

clean-lint:
	@rm -rf ./flake8_report.txt ./flake8_report_junit.xml

clean-setup:
	@rm .env

lint:
	@make clean-lint
	@flake8 ingenii_quantum --tee --output-file flake8_report.txt

lint-convert:
	@flake8_junit flake8_report.txt flake8_report_junit.xml

build:
	@make lint
	@if [ -d build ]; then rm -r build; fi
	@if [ -d dist ]; then rm -r dist; fi
	python setup.py sdist bdist_wheel

check:
	twine check dist/*

upload: check
	twine upload --config-file .pypirc dist/*

upload-test: check
	twine upload --repository testpypi --config-file .pypirc dist/*
