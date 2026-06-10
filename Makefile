.PHONY: install test demo convert clean

install:
	pip install -r requirements.txt

test:
	pytest -q

demo:
	python -m pipeline demo

convert:
	python -m pipeline convert mobilenet_v3_small Model.mlpackage
	python -m pipeline optimize Model.mlpackage Model_int8.mlpackage --dtype int8
	python -m pipeline inspect Model_int8.mlpackage

clean:
	rm -rf *.mlpackage Model*.mlpackage .pytest_cache __pycache__ */__pycache__ */*/__pycache__
