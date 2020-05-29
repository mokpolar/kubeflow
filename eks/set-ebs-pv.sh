VOLUME_ID=$(aws ec2 create-volume --size 50 --region us-east-1 --availability-zone us-east-1c --volume-type gp2 | jq '.VolumeId' -)
cat ebs-pv.yaml | sed "s/<VOLUME_ID>/$VOLUME_ID/g" | kubectl apply -f -




