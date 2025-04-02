#!/bin/bash
export RELEASE=1

pushd ./transformer
make clean
make
if [ $? -ne 0 ]; then
    echo "transformer build failed"
    exit 1
fi
popd
