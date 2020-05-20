import uuid
from kubeflow import fairing
from kubeflow.fairing.kubernetes import utils as k8s_utils

CONTAINER_REGISTRY = 'mokpolar'
namespace = 'admin'
job_name = f'tensorflow-mnist-job-{uuid.uuid4().hex[:4]}'
command = ["python", "seldon-tensorflow_mnist.py", "--model_path", "/mnt/pv/tensorflow/mnist/models"]
output_map = {
    "Dockerfile": "Dockerfile",
    "seldon-tensorflow_mnist.py": "seldon-tensorflow_mnist.py"
}
fairing.config.set_preprocessor('python', command=command, path_prefix="/app", output_map=output_map)
fairing.config.set_builder('docker', registry=CONTAINER_REGISTRY, image_name="tensorflow-mnist",
                           dockerfile_path="Dockerfile")
fairing.config.set_deployer('job', namespace=namespace, job_name=job_name,
                            pod_spec_mutators=[
                                k8s_utils.mounting_pvc(pvc_name='vol-my-pvc2', pvc_mount_path='/mnt/pv')],
                            cleanup=True, stream_log=True)
fairing.config.run()