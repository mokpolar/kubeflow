VOLUME_ID=$(aws ec2 create-volume --size 20 --region us-east-1 --availability-zone us-east-1a --volume-type gp2 | jq '.VolumeId' -)
cat ./kafka-ebs.yaml | sed "s/<VOLUME_ID>/$VOLUME_ID/g" | kubectl apply -f -




