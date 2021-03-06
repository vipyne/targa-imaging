# targa imaging
![new alt](https://github.com/vipyne/targa-imaging/blob/master/readme.png)

Command line tool to "translate" any file type into a [TARGA file](https://en.wikipedia.org/wiki/Truevision_TGA).

compile : (anything in the `works` directory)
```
$ gcc works/d.c -o targa-exe
```

usage   :
```
$ ./targa-exe input-file output-filename dimension
```

example :
```
$ ./targa-exe /usr/input/filename.txt /usr/output/filename$(date +%s).tga 1000
```

yields => a targa image `filename.tga` that is 1000 x 1000 pixels large

Basically, if you like glitchart and archaic image formats, then you'll love this.

## copy a targa and make it a larger

compile :
```
$ gcc works/copy-tga.c -o targa-enhance.exe
```

usage   :
```
$ ./targa-enhance.exe original-square-targa-file output-filename original-dimension
```

example :
```
$ ./targa-enhance.exe /usr/input/og.tga /usr/output/larger-targa-enhance.exe.tga 1000
```

## using CUDA 7.5 framework on OSX (wip)

[download CUDA 7.5](https://developer.nvidia.com/cuda-75-downloads-archive) and follow instructions.

`$ brew install clang++-3.7`
(this avoids the fun `nvcc fatal   : The version ('80000') of the host compiler ('Apple clang') is not supported` error)

compile :
```
$ nvcc create-targa.cu -ccbin=$(which clang++-3.7) -o targa-exe
```

usage   :
```
$ ./targa-exe input-file output-filename dimension cuda
```


