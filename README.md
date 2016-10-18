# targa-imaging

input any file type - and "translate" it into a targa image.

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
          
example yields ==> a targa image 'filename.tga' that is 1000 x 1000 pixels large
