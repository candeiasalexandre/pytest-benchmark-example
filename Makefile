.PHONY: clean-env
clean-env:
	-rm -rf .venv

.PHONY: setup-dev
setup-dev: clean
	poetry config virtualenvs.in-project true 
	poetry install

.PHONY: test
test:
	poetry run pytest tests

.PHONY: simple-benchmark
simple-benchmark:
	poetry run pytest tests/test_model.py::test_benchmark_model_batch_1


.PHONY: parametrized-benchmark
parametrized-benchmark:
	poetry run pytest tests/test_model.py::test_benchmark_model_parametrized --benchmark-histogram=output/histogram_example