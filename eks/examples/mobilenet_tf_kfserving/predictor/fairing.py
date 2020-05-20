import uuid
from kubeflow  import fairing # 이게 있어야지
from kubeflow.fairing.kubernetes import utils as k8s_utils

# user variable

CONTAINER_REGISTRY = "mokpolar"
namespace = "mobilenet"
job_name = f"tf-job-{uuid.uuid4().hex[:4]}"


# command

command = ["python" , "keras_mobnet.py", "--model_path", "/mnt/pv/models/keras/mobnet"]

output_map = {
    "Dockerfile": "Dockerfile",
    "keras_mobnet.py": "keras_mobnet.py"
}

# fairing

# preprocessor : when you build container image, defines informations for image. 
fairing.config.set_preprocessor("python", command=command, path_prefix="/app", output_map=output_map)

# build : how to build container image. where to. 
fairing.config.set_builder("docker", registry=CONTAINER_REGISTRY, image_name="kfserving-mobilenet", dockerfile_path="Dockerfile")

# depooyer : deploy image. implementation(where) . 
# job : kubernetes job resource
fairing.config.set_deployer("job", namespace=namespace, job_name=job_name, pod_spec_mutators=[k8s_utils.mounting_pvc(pvc_name="my-pvc", pvc_mount_path="/mnt/pv")], \
        cleanup=True, stream_log=True) # cleanup이 뭐더라



fairing.config.run()