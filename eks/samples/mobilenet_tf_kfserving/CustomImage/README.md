
# README

This test was conducted to incorporate the preprocessing code used in the transformer into the Predictor without using the transformer.


## Process

- model.py exists. 

1. Custom Predictor Image build
```sh
docker build {USERS}/{DOCKER}:{TAG} . 
```

2. Push
```sh
docker push {USERS}/{DOCKER}:{TAG}
```

3. Deploy
```sh
kubectl apply custom_deploy.yaml -n {NAMESPACE}
```

4. Inference
```sh
kubectl -
```
