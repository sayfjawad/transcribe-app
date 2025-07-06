import nemo.collections.asr as nemo_asr

# List available models
available_models = nemo_asr.models.EncDecCTCModel.list_available_models()

# Print details of each model
for model in available_models:
    print(model.pretrained_model_name)  # Use dot notation
    print(model.description)
    print(model.location)  # Some versions use .location instead of .path
