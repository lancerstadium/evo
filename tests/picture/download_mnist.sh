#!/bin/bash

# 定义文件名和目录名
MNIST_URL_BASE="http://yann.lecun.com/exdb/mnist/"
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
        echo "$FILE already exists, skipping download..."
    else
        echo "Downloading $FILE..."
        wget --no-check-certificate -q --show-progress "$MNIST_URL_BASE$FILE" -O "$FILE_PATH"
        
        # 检查下载是否成功
        if [ $? -ne 0 ] || [ ! -s "$FILE_PATH" ]; then
            echo "Failed to download $FILE. Please check your internet connection or the URL."
            rm -f "$FILE_PATH"
            continue
        fi
    fi
    
    # 解压文件
    echo "Extracting $FILE..."
    gzip -d "$FILE_PATH"
    
    # 检查解压是否成功
    if [ $? -ne 0 ]; then
        echo "Failed to extract $FILE with gzip. Attempting with zcat..."
        zcat "$FILE_PATH" > "$DIRECTORY/$UNZIPPED_FILE"
        
        # 检查zcat是否成功
        if [ $? -ne 0 ]; then
            echo "Failed to extract $FILE with zcat. The file might be corrupted."
            rm -f "$DIRECTORY/$UNZIPPED_FILE"
            continue
        fi
    fi
    
    # 确认解压后的文件是否存在
    if [ ! -f "$DIRECTORY/$UNZIPPED_FILE" ]; then
        echo "Extraction of $FILE failed. The file might be corrupted."
        continue
    fi
    
    echo "$FILE extracted successfully."
done

echo "MNIST dataset is ready in $DIRECTORY."
