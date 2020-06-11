import uuid
from kubeflow import fairing
from kubeflow.fairing.kubernetes import utils as k8s_utils

CONTAINER_REGISTRY = 'mokpolar'

namespace = 'mobilenet'
job_name = f'sklearn-iris-job-{uuid.uuid4().hex[:4]}'

command=["python", "pytorch_cifar10.py", "--model_path", "/mnt/pv/models/pytorch/cifar10"]
output_map = {
    "Dockerfile": "Dockerfile",
    "pytorch_cifar10.py": "pytorch_cifar10.py"
}

fairing.config.set_preprocessor('python', command=command, path_prefix="/app", output_map=output_map)

fairing.config.set_builder('docker', registry=CONTAINER_REGISTRY, image_name="pytorch-cifar10", dockerfile_path="Dockerfile")

fairing.config.set_deployer('job', namespace=namespace, job_name=job_name,
                            pod_spec_mutators=[k8s_utils.mounting_pvc(pvc_name='my-pvc', pvc_mount_path='/mnt/pv')],
                            cleanup=True, stream_log=True)

fairing.config.run()
