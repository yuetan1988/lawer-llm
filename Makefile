INFERENCE_HOST=0.0.0.0
INFERENCE_PORT=8000


start:  ## Start the inference server
	@echo "Starting the inference server"
	uvicorn app:app --host "$(INFERENCE_HOST)" --port "$(INFERENCE_PORT)"
