VOLUME_ID=$(aws ec2 create-volume --size 35 --region us-east-1 --availability-zone us-east-1b --volume-type gp2 | jq '.VolumeId' -)
cat ./pv.yaml | sed "s/<VOLUME_ID>/$VOLUME_ID/g" | kubectl apply -f -




