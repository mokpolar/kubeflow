# KFServing with MobileNet on AWS
AWS EKS Cluster 상에 Kubeflow 환경을 구성하였고,   
가벼운 이미지 처리 모델(MobileNet)을 KFServing을 이용하여 Serving 하는 작업입니다.  
Serving된 모델로 들어오기 전에 KFServing의 Transformer를 이용해 전처리가 수행됩니다. 

## Process
* Set environment using [Makefile](https://github.com/mokpolar/kubeflow/blob/master/eks/Makefile)
    * Create eks cluster & nodegroup
    * Install kubeflow on nodegroup
```py
make create_cluster
make create_nodegroup
make install_kubeflow
```
* kfserving-ingressgateway -> istio-ingressgateway
```bash
kubectl -n knative-serving apply -f ./base-settings/config-istio.yaml
kubectl -n kubeflow apply -f ./base-settings/inferenceservice-config.yaml
kubectl -n knative-serving apply -f ./base-settings/knative-ingress-gateway.yaml
```

* Build PV on EBS
* Bind PVC

* Fairing (predictor)
    * model saving file
    * docker file
    * job implementation (job : kubernets resouce)

* Deploy infernece service

* Fairing (transformer)
    * preprocessing file
    * docker file
    * job implementation

* Deploy transformer

---
