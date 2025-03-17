.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y bgg_project || :
	@pip install -e .


run main:
	python -c 'from bgg_project.interface.main import preprocess_and_train; preprocess_and_train()'
