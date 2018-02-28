test:
	green -vvv

test-examples:
	python -m examples.example_ecc \
	&& python -m examples.example_br \
	&& python -m examples.example_lp \
	&& python -m examples.example_cc \
	&& python -m examples.example_pcc

.PHONY: test
