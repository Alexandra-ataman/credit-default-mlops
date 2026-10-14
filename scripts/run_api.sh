#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
docker build -t credit-default .
docker run --rm -p 127.0.0.1:8000:8000 credit-default
