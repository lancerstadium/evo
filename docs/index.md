---
title: "EVO"
author: LancerStadium
data: July 7, 2024
output: pdf_document
presentation:
    enableSpeakerNotes: true
    theme: serif.css
    width: 960
    height: 700
    slideNumber: true
    progress: false

---


# EVO


## 1 Route


|  case  | doc | ppt |
|:------:|:---:|:---:|
|  model |     |     |



## 2 Demo

```c
int main() {
    printf("hello world!\n");
    return 0;
}
```

## 3 Statistic

Code scale:

```make {cmd=true args=["-f", "$input_file", "-s", "line"]}
line:
	@wc -l `find ../ -name "*.c";find -name "*.h"`
```