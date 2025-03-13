.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
        @pip uninstall -y bgg || :
        @pip install -e .
