# Deploy the model to Azure
from azureml.core.model import Model
from azureml.core.webservice import AciWebservice
from azureml.core.webservice import Webservice
from azureml.core.model import InferenceConfig

# Register the model
model = Model.register(workspace=ws, model_path='ml_model/tf_model.h5', model_name='investment-advisor-model')

# Define inference configuration
inference_config = InferenceConfig(entry_script='api.py', environment=env)

# Define deployment configuration
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Deploy the model
service = Model.deploy(workspace=ws, name='investment-advisor-service', models=[model], inference_config=inference_config, deployment_config=deployment_config, deployment_target=compute_target)
service.wait_for_deployment(show_output=True)

# Get the web service endpoint
print(service.scoring_uri)
