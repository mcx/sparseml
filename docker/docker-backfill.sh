#!/usr/bin/env bash

export CR_PAT=XXXXXXX ; echo $CR_PAT | docker login ghcr.io -u <user-name> --password-stdin

# base sparseml 1.5.0
DOCKER_BUILDKIT=1 \
  docker build  --build-arg DEPS=base \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --build-arg VERSION=1.5 \
  -t ghcr.io/neuralmagic/sparseml-base:1.5 .

docker push ghcr.io/neuralmagic/sparseml-base:1.5



# all sparseml 1.5.0
DOCKER_BUILDKIT=1 \
  docker build  --build-arg DEPS=all \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --build-arg VERSION=1.5 \
  -t ghcr.io/neuralmagic/sparseml:1.5 .

docker push ghcr.io/neuralmagic/sparseml:1.5


# nightly sparseml latest
DOCKER_BUILDKIT=1 \
  docker build  --build-arg DEPS=all \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --build-arg MODE=nightly \
  -t ghcr.io/neuralmagic/sparseml-nightly:latest .

docker push ghcr.io/neuralmagic/sparseml-nightly:latest