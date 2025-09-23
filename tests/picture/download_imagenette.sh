#!/bin/bash

# 定义文件名和目录名
SUFFIX_SIZE="-320"  # "": original size, "-320": 320x320, "-160": 160x160
IMAGENETTE_URL_BASE="https://s3.amazonaws.com/fast-ai-imageclas/"
FILES=("imagenette2$SUFFIX_SIZE.tgz")
DIRECTORY="imagenette"

# 创建目标目录
if [ ! -d "$DIRECTORY" ]; then
    echo "Creating directory $DIRECTORY..."
    mkdir $DIRECTORY
fi


# 下载并解压文件
for FILE in "${FILES[@]}"; do

    # 检查解压前的文件是否已经存在
    if [ -f "$DIRECTORY/$FILE" ]; then
        echo "$DIRECTORY/$FILE already exists. Skipping download..."
    else 
        echo "Downloading $FILE..."
        wget -O "$DIRECTORY/$FILE" "$IMAGENETTE_URL_BASE$FILE"
    fi

    echo "Extracting $FILE..."
    tar -xzf "$DIRECTORY/$FILE" -C "$DIRECTORY"
    rm "$DIRECTORY/$FILE"
done

echo "Imagenette dataset is ready in $DIRECTORY."