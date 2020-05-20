.PHONY: create_cluster create_nodegroup install_kubeflow del_cluster del_nodegroup uninstall_kubeflow

# USAGE
## 1. when you do not have eks cluster
### $make create_cluster
## 2. create nodegroup in cluster
### $make create_nodegroup
## 3. kubeflow setup
### $make kubeflow_setup
## 4. delete nodegroup 
### $make del_nodegroup
## 5. delete eks cluster
### $make del_eks

# user variables. you should change these variables. 
cluster_name = jyj-cluster-test-1
node_name = jyj-node-2
node_type = m5.xlarge # AL2_x86_64
gpu_node_type = p3.2xlarge # AL2_x86_64_GPU
ssh_key_pair = jungjuyoung-dist-test


# default env
base_dir = ./kube_deployment
kf_dir = ${base_dir}/${cluster_name}

# kubeflow manifest for aws  1.0.2
config_uri = "https://raw.githubusercontent.com/kubeflow/manifests/v1.0-branch/kfdef/kfctl_aws.v1.0.2.yaml" # latest file

# kubeflow manifest for aws 1.0.2 for cognito
cognito_uri = "https://raw.githubusercontent.com/kubeflow/manifests/v1.0-branch/kfdef/kfctl_aws_cognito.v1.0.2.yaml"


# user variables. changes are not required.
region = us-east-1
#node_iam = jyjung-eks-cluster-node-policy1

# kubernetes version
kube_ver = 1.15

# kfctl 1.0.2 version
kfctl_file = https://github.com/kubeflow/kfctl/releases/download/v1.0.2/kfctl_v1.0.2-0-ga476281_darwin.tar.gz



create_cluster:
	# create cluster
	eksctl create cluster --name ${cluster_name} --region ${region}
	# eks config set
	aws eks --region ${region} update-kubeconfig --name ${cluster_name}
	# check eks cluster, nodes. 
	kubectl get svc
	eksctl get nodegroups --cluster ${cluster_name}

create_nodegroup:
	# eks config set
	aws eks --region ${region} update-kubeconfig --name ${cluster_name}
	# create managed nodegroup 
	eksctl create nodegroup --cluster ${cluster_name} --version ${kube_ver} --name ${node_name} --node-type ${gpu_node_type} \
	--nodes 2 --nodes-min 2 --nodes-max 4 --node-volume-size 50 --ssh-public-key ${ssh_key_pair} --managed

install_kubeflow:
	# eks config set
	aws eks --region ${region} update-kubeconfig --name ${cluster_name}

	# download kfctl
	curl -LO ${kfctl_file}

	# unzip kfctl
	tar -xvf kfctl_v1.0.2-0-ga476281_darwin.tar.gz

	# make kubeflow deployment directory
	mkdir -p ${kf_dir}

	#cd ${kf_dir} 

	# config_uri or cognito_uri
	wget -O ${kf_dir}/kfctl_aws.yaml ${config_uri}

	# change config file
	# use IAM for Service Account
	sed -i -e 's/region: us-west-2/region: us-east-1/g' ${kf_dir}/kfctl_aws.yaml
	sed -i -e 's/roles:/enablePodIamPolicy: true/g' ${kf_dir}/kfctl_aws.yaml
	sed -i -e '/eksctl/d' ${kf_dir}/kfctl_aws.yaml

	# deploy kubeflow
	#./kfctl apply -V -f ${kf_dir}/kfctl_aws.yaml
	#kubectl -n kubeflow get all

uninstall_kubeflow:
	cd ${kf_dir}
	kfctl delelte -f ${kf_dir}/kfctl_aws.yaml


del_nodegroup:
	eksctl delete nodegroup --cluster ${cluster_name} --name ${node_name}

del_cluster:
	eksctl delete cluster --name ${cluster_name}


