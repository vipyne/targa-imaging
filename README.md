# targa imaging

Command line tool to "translate" any file type into a [TARGA file](https://en.wikipedia.org/wiki/Truevision_TGA).

compile : 
```
$ gcc create-tga-from-any-input.c -o targa-exe
```

usage   : 
```
$ ./targa-exe input-file output-filename dimension
```

example : 
```
$ ./targa-exe /usr/input/filename.txt /usr/output/filename.tga 1000
```
          
yields => a targa image `filename.tga` that is 1000 x 1000 pixels large

Basically, if you like glitchart and archaic image formats, than you'll love this.

![new alt](https://github.com/vipyne/targa-imaging/blob/master/readme.png) 
