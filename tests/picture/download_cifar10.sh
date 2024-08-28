#!/bin/bash

# 定义文件名和目录名
TARBALL="cifar-10-binary.tar.gz"
DIRECTORY="cifar10"
URL="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"

# 检查是否已经下载了数据集的压缩文件或存在文件夹
if [[ -f "$TARBALL" || -d "$DIRECTORY" ]]; then
    echo "$TARBALL or $DIRECTORY already exists, skipping download."
else
    echo "Downloading $TARBALL..."
    wget $URL -O $TARBALL
fi

# 检查是否已经解压了数据集
if [ -d "$DIRECTORY" ]; then
    echo "Directory $DIRECTORY already exists, skipping extraction."
else
    echo "Extracting $TARBALL..."
    tar xvzf $TARBALL
    mv cifar-10-batches-bin $DIRECTORY
fi

# 删除压缩文件和临时解压的目录（如果存在）
if [ -f "$TARBALL" ]; then
    echo "Removing $TARBALL..."
    rm -f $TARBALL
fi

if [ -d "cifar-10-batches-bin" ]; then
    echo "Removing temporary directory cifar-10-batches-bin..."
    rm -rf cifar-10-batches-bin
fi

echo "Dataset is ready in $DIRECTORY."
