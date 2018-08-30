// vanessa writes a targa file
// compiling and running this program will produce a targa file
//
// compile : $ gcc copy_a_tga.c -o targa-enhance.exe
// usage   : $ ./targa-enhance.exe input-tga output-filename original-targa-dimensions
// example : $ ./targa-enhance.exe /usr/input/original.tga /usr/output/filename.tga 1000
//           ==> filename.tga that is 1000 x 1000 pixels large
//
// a lot of this is based on Grant Emery's file https://www.tjhsst.edu/~dhyatt/superap/code/targa.c thanks dude
// author: vanessa pyne --- github.com/vipyne



#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <math.h>

#define BYTE_RANGE 256
#define RGBA 3 // 3 for RGB, 4 for RGBA

////// targa file header

typedef struct {
char id_length;       // length of id field (number of bytes - max 255)
char map_type;        // colormap field (0 or 1; no map or 256 entry palette)
char image_type;      // ( 0 - no image data included
                      //   1 - uncompressed, color mapped image
                      //   2 - uncompressed, RGB image
                      //   3 - uncompressed, black & white image
                      //   9 - run-length encoded(RLE-lossless compression),color mapped image
                      //  10 - RLE, RGB image
                      //  11 - compressed, black & white image )

  int map_first;       // first entry index for color map
  int map_length;      // total number of entries in color map
  char map_entry_size; // number of bits per entry

  int x;               // x cooridinate of origin
  int y;               // y cooridinate of origin

  int width;           // width in pixels
  int height;          // height in pixels

  char bits_per_pixel; // number of bits per pixel

  char misc;           // srsly? "scan origin and alpha bits" this example uses scan origin
                       // honestly, don't know what's going on here. we pass in a hex value
                       // :shrug_emoji:
} targa_header;

int little_endianify (int number)
{
  return number % BYTE_RANGE;
}

int big_endianify (int number)
{
  return number / BYTE_RANGE;
}

// used if sorting the pixels
int compare_function (const void* a_pointer, const void* b_pointer)
{
  return *(( char* )a_pointer) - *(( char* )b_pointer);
}

////// write header function

void write_header (targa_header header, FILE *tga)
{
  fputc( header.id_length, tga );
  fputc( header.map_type, tga );
  fputc( header.image_type, tga );

  fputc( little_endianify(header.map_first), tga );
  fputc( big_endianify(header.map_first), tga );

  fputc( little_endianify(header.map_length), tga );
  fputc( big_endianify(header.map_length), tga );

  fputc( header.map_entry_size, tga );

  fputc( little_endianify(header.x), tga );
  fputc( big_endianify(header.x), tga );
  fputc( little_endianify(header.y), tga );
  fputc( big_endianify(header.y), tga );

  fputc( little_endianify(header.width), tga );
  fputc( big_endianify(header.width), tga );
  fputc( little_endianify(header.height), tga );
  fputc( big_endianify(header.height), tga );

  fputc( header.bits_per_pixel, tga );
  fputc( header.misc, tga );
}

void print_directions(void)
{
  printf("$ ./targa-exe input-file output-filename dimension\n");
}

////// MAIN

int main (int argc, char* argv[])
{
  printf("^^^^ targa_header, length: %d \n", (char) sizeof(targa_header)); // 36

  if (argc != 5) {
    printf("\n");
    printf("Please enter correct number of arguments. --\n");
    print_directions();
    printf("\n");
    return 1;
  }

  FILE *source;
  source = fopen(argv[1], "rb");

  if (source == NULL) {
    printf("Source file `%s` cannot be found. --\n", argv[1]);
    return 1;
  }

  FILE *tga;                    // pointer to file that we will write
  tga = fopen(argv[2], "wbx");  // `x` needed for `errno` to work
  int overwrite_warning = errno;

  if (overwrite_warning != 0) {
    printf("Destination file `%s` already exists. --\n", argv[2]);
    return 1;
  }

  int HEIGHT = atoi(argv[3]);
  int WIDTH = atoi(argv[3]);
  int FACTOR = atoi(argv[4]);

  if (errno != 0) {
    fclose(tga);
    unlink(argv[2]);

    printf("`%s` is not a valid dimension. Please use a number. --\n", argv[3]);
    return 1;
  }


  // intialize and set TARGA header values
  targa_header header;       // variable of targa_header type

  int x, y;                  // coordinates for `for` loops to pass in
                             // correct number of pixel values

  header.id_length = 0;
  header.map_type = 0;
  header.image_type = 2;     // uncompressed RGB image

  header.map_first = 0;
  header.map_length = 0;
  header.map_entry_size = 0;

  header.x = 0;
  header.y = 0;
  // header.width = WIDTH;
  // header.height = HEIGHT;
  // int factor = 3;

  int initial_og_size = 1000;

  header.width = initial_og_size * FACTOR;  ////////////////////////////////////////
  header.height = initial_og_size * FACTOR; ////////////////////////////////////////

  header.bits_per_pixel = 24;
  header.misc = 0x20;       // scan from upper left corner, need to investigate this further

  // start to write file
  write_header(header, tga);
  printf("^^^^ header written\n");

  // source input file

  fseek(source, 0, SEEK_END);
  int source_size = ftell(source);
  rewind(source);
  printf("^^^^ source file read,     d,  length: %d\n", source_size);
  printf("^^^^ sizeof(targa_header) lu,  length: %lu\n", sizeof(targa_header));

  fseek(source, 18 * sizeof(char), SEEK_SET);
  int header_size = ftell(source);
  printf("^^^^ targa header,         d,  length: %d\n", header_size);

  char *source_buffer = (char*)malloc(sizeof(char) * (source_size - header_size));

  // rewind(source);
  size_t read = fread(source_buffer, sizeof(char), source_size, source);
  printf("^^^^ read lu,  length: %lu\n", read);

  int column = 0;

  // write column
  for (int y = 0; y < initial_og_size; y++)
  {
    int row_i = (column * initial_og_size * RGBA);
    for (int ff = 0; ff < FACTOR; ff++)
    {
      // write row
      for (int x = 0; x < initial_og_size; x++)
      {
        for (int f = 0; f < FACTOR; f++)
        { /// factor
          // write pixel
          for (int rgb = 0; rgb < 1; rgb++) {
            fputc(source_buffer[row_i], tga);
            fputc(source_buffer[row_i+1], tga);
            fputc(source_buffer[row_i+2], tga);
          }
        }
        row_i+=RGBA;
      }
    }
    column++;
  }

  fclose(tga);
  fclose(source);
  printf("^^^^ finished! marvel at your targa!\n");

  return 0;
}
