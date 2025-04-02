#!/bin/bash

DOCKER_IMAGE=rb
version=$1
size=$2
docker build -t ${DOCKER_IMAGE} docker/
{ time \
      docker run --net=host -v $(pwd):/home ${DOCKER_IMAGE} python3 \
      main.py --version ${version} --n ${size} --generate-config
} 2>${project}.duration
