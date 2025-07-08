## 一个好用的镜像源
-i http://pypi.mirrors.ustc.edu.cn/simple --trusted-host pypi.mirrors.ustc.edu.cn

## 二进制零基础学习：
B站大学搜 liveoverflow(一个著名黑客）、Youtube上有50多节，b站上只有41节
十分钟的视频看个一两个小时很正常

## 栈溢出

- [栈介绍](https://ctf-wiki.org/pwn/linux/user-mode/stackoverflow/x86/stack-intro/)
- [栈帧结构](https://www.cnblogs.com/clover-toeic/p/3755401.html)
- [栈溢出](https://ctf-wiki.org/pwn/linux/user-mode/stackoverflow/x86/stackoverflow-basic/)

## Huggingface连接问题

```
OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like hfl/chinese-roberta-wwm-ext is not the path to a directory containing a file named config.json.
Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
```

```bash
export HF_ENDPOINT=https://hf-mirror.com
```
