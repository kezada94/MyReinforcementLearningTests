START_PIPELINE_PATH := utils
START_PIPELINE_EXEC := start_pipeline.py

RUN_ID := $(shell python ${START_PIPELINE_PATH}/${START_PIPELINE_EXEC} ${RUN_NAME})

.PHONY: run
run:
	export MLFLOW_RUN_ID=${RUN_ID}; \
	echo $$MLFLOW_RUN_ID; \
	dvc repro