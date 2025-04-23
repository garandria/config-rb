#!/bin/bash

DOCKER_IMAGE=rb
version=$1
size=$2
output=$3
project=$4
docker build -t ${DOCKER_IMAGE} .
{ time \
      docker run --net=host -v $(pwd):/home ${DOCKER_IMAGE} python3 \
      main.py --version ${version} --n ${size} --generate-config --output ${output}
} 2>${project}.duration
