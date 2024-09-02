#!/bin/bash

# 使用可用的镜像链接
MNIST_URL_BASE="https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES=("train-images-idx3-ubyte.gz" "train-labels-idx1-ubyte.gz" "t10k-images-idx3-ubyte.gz" "t10k-labels-idx1-ubyte.gz")
DIRECTORY="mnist"

# 创建目标目录
if [ ! -d "$DIRECTORY" ]; then
    echo "Creating directory $DIRECTORY..."
    mkdir -p $DIRECTORY
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
    
    # 检查并下载文件
    if [ -f "$FILE_PATH" ]; then
        echo "$FILE already exists, checking size..."
        if [ ! -s "$FILE_PATH" ]; then
            echo "$FILE is empty, re-downloading..."
            rm -f "$FILE_PATH"
        else
            echo "$FILE is not empty, skipping download..."
            continue
        fi
    fi
    
    # 下载文件，使用curl尝试，如果失败则使用wget
    echo "Downloading $FILE..."
    curl -o "$FILE_PATH" "$MNIST_URL_BASE$FILE" --retry 3 --retry-delay 5
    if [ $? -ne 0 ]; then
        echo "curl failed, trying wget..."
        wget "$MNIST_URL_BASE$FILE" -O "$FILE_PATH"
        if [ $? -ne 0 ]; then
            echo "Failed to download $FILE with both curl and wget."
            continue
        fi
    fi
    
    # 检查文件大小是否为0
    if [ ! -s "$FILE_PATH" ]; then
        echo "Downloaded file $FILE is empty, skipping extraction."
        continue
    fi
    
    # 解压文件
    echo "Extracting $FILE with gzip..."
    gzip -d "$FILE_PATH"
    
    # 检查解压是否成功
    if [ $? -ne 0 ]; then
        echo "Failed to extract $FILE with gzip. Attempting with zcat..."
        zcat "$FILE_PATH" > "$DIRECTORY/$UNZIPPED_FILE"
        if [ $? -ne 0 ]; then
            echo "Failed to extract $FILE with zcat."
        fi
    fi
done

echo "MNIST dataset is ready in $DIRECTORY."
