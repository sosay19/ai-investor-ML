# Setup Azure Machine Learning
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core.compute import ComputeTarget, AmlCompute

# Connect to your workspace
ws = Workspace.from_config()

# Create a new experiment
experiment = Experiment(workspace=ws, name='investment-advisor-experiment')

# Define the environment
env = Environment(name='tf-env')
env.python.conda_dependencies.add_pip_package('tensorflow')
env.python.conda_dependencies.add_pip_package('flask')

# Create a compute target
compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS2_V2', max_nodes=1)
compute_target = ComputeTarget.create(ws, 'my-compute-cluster', compute_config)
compute_target.wait_for_completion(show_output=True)
