#!/bin/bash

# 定义文件名和目录名
MNIST_URL_BASE="http://yann.lecun.com/exdb/mnist/"
FILES=("train-images-idx3-ubyte.gz" "train-labels-idx1-ubyte.gz" "t10k-images-idx3-ubyte.gz" "t10k-labels-idx1-ubyte.gz")
DIRECTORY="mnist"

# 创建目标目录
if [ ! -d "$DIRECTORY" ]; then
    echo "Creating directory $DIRECTORY..."
    mkdir $DIRECTORY
fi

# 下载并解压文件
for FILE in "${FILES[@]}"; do
    FILE_PATH="$DIRECTORY/$FILE"
    UNZIPPED_FILE="${FILE%.gz}"
    
    # 检查解压后的文件是否存在
    if [ -f "$DIRECTORY/$UNZIPPED_FILE" ]; then
        echo "$UNZIPPED_FILE already exists, skipping download and extraction."
        continue
    fi
    
    # 下载文件
    echo "Downloading $FILE..."
    wget "$MNIST_URL_BASE$FILE" -O "$FILE_PATH"
    
    # 解压文件
    echo "Extracting $FILE with gzip..."
    gzip -d "$FILE_PATH"
    
    # 检查解压是否成功
    if [ $? -ne 0 ]; then
        echo "Failed to extract $FILE with gzip. Attempting with zcat..."
        zcat "$FILE_PATH" > "$DIRECTORY/$UNZIPPED_FILE"
    fi
done

echo "MNIST dataset is ready in $DIRECTORY."
