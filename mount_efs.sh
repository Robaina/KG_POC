#!/bin/bash

mkdir -p /home/ec2-user/SageMaker/efs/sandbox/
mkdir -p /home/ec2-user/SageMaker/efs/datasets/


sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 fs-062b164e3eb5ab460.efs.us-west-2.amazonaws.com:/ /home/ec2-user/SageMaker/efs/sandbox/
sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,ro fs-0ae2ebeb78e658723.efs.us-west-2.amazonaws.com:/ /home/ec2-user/SageMaker/efs/datasets/