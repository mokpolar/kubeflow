## 테스트 목표
* Kubeflow KFServing에 탑재된 모델을 이용해 이미지 처리시 Latency 향상을 위해 이미지 전송 방식 별로 이를 측정한다.  

## 테스트 시나리오
* Kubernetes Cluster 위에 KFServing을 구성하고 이미지 처리 모델을 탑재
* K8S Cluster와 같은 데이터 센터에 위치한 서버를 별도 구성 후 해당 서버에서 KFSerivng에 이미지 전송
* 이미지 전송 방식 : Curl, Kafka
* 측정 항목 : 구간별 도착 및 걸린 시간(Unix Time)
## 테스트 조건
### 환경 구성
* AWS us-east-1
* Curl ![architecture](https://github.com/mokpolar/kubeflow/tree/master/eks/samples/kafka/images/arch2.png)
* Kafka ![architecture](https://github.com/mokpolar/kubeflow/tree/master/eks/samples/kafka/images/arch1.png)
* 전송 대상 이미지
    * JPG 파일 > JSON 변환 (변환 시 사이즈 변화)
        * 1.1MB > 1.5MB
        * 3.3MB > 4.6MB
        * 4.7mB > 6.4MB
        * 10.2MB > 13.9MB

## 테스트

* 테스트 프로그램

## 테스트 결과 및 적용 시 고려사항

