import uuid
from kubeflow  import fairing
from kubeflow.fairing.kubernetes import utils as k8s_utils

# user variable

CONTAINER_REGISTRY = "mokpolar"

namespace = "mobilenet"
job_name = f"tf-job-{uuid.uuid4().hex[:4]}"


# command

command = ["python" , "keras-mobilenet.py", "--model_path", "/mnt/pv/models/tensorflow/mobilenet"]

output_map = {
    "Dockerfile": "Dockerfile",
    "keras-mobilenet.py": "keras-mobilenet.py"
}

# fairing

# preprocessor : when you build container image, defines informations for image. 
fairing.config.set_preprocessor("python", command=command, path_prefix="/app", output_map=output_map)

# build : how to build container image. where to. 
fairing.config.set_builder("docker", registry=CONTAINER_REGISTRY, image_name="keras-mobilenet", dockerfile_path="Dockerfile")

# depooyer : deploy image. implementation(where) . 
# job : kubernetes job resource
fairing.config.set_deployer("job", namespace=namespace, job_name=job_name, pod_spec_mutators=[k8s_utils.mounting_pvc(pvc_name="my-pvc", pvc_mount_path="/mnt/pv")], cleanup=True, stream_log=True)



fairing.config.run()