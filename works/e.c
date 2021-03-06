// vanessa writes a targa file
// compiling and running this program will produce a targa file
//
// compile : $ gcc create-tga-from-any-input.c -o targa-exe
// usage   : $ ./targa-exe input-file output-filename dimensions
// example : $ ./targa-exe /usr/input/filename.txt /usr/output/filename.tga 1000
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
  if (argc != 4) {
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
  header.width = WIDTH;
  header.height = HEIGHT;

  header.bits_per_pixel = 24;
  header.misc = 0x20;       // scan from upper left corner, need to investigate this further

  // start to write file
  write_header(header, tga);
  printf("^^^^ header written\n");

  // source input file

  fseek(source, 0, SEEK_END);
  int source_size = ftell(source);
  rewind(source);
  printf("^^^^ source file read,      length: %d\n", source_size);

  // width * height = number of pixels
  int input_binary_length = WIDTH * HEIGHT * RGBA; // normal people call this a buffer

  // buffer for pixel values (no zeros)
  char normalized_input[input_binary_length];

  // buffer for sorted pixel values (still no zeros)
  char normalized_sorted[input_binary_length];

  // buffer for entire input file
  char *read_through = (char*) malloc ( sizeof(char) * source_size );

  int i = 0;
  int read_through_index = 0;
  while (i < input_binary_length)
  {
    fread(read_through, 1, source_size, source);
    if (read_through_index >= source_size)
    {
      rewind(source);
      read_through_index = 0;
    }
    if (read_through[read_through_index] != '0')
    {
      normalized_input[i] = read_through[read_through_index];
      i++;
    }
    read_through_index++;
  }
  free(read_through);
  printf("^^^^ normalized buffer set, length: %d \n", (int) sizeof(normalized_input));

  strncpy(normalized_sorted, normalized_input, input_binary_length);
  qsort(normalized_sorted, strlen(normalized_input), sizeof(char), compare_function);

  printf("^^^^ writing pixels \n");

  int tga_header_size = ftell(tga);
  int tga_current_position = ftell(tga);;
  int pixel_count = 0;
  int n_index = 0;
	float theta = 0;
  //// magic happens here
  for (int y = 0; y < HEIGHT; ++y)
  {

		int line = (float)cos(theta/10.0)*1 + 10*x;

    for (int x = 0; x < WIDTH && (pixel_count <= input_binary_length); ++x)
    {


      float dada = log(theta) * 70.0;

      // BLUE //

      if (x % 2 ==0) {

        if (y > line+ normalized_input[n_index - 1] || (float)log(y/100.0) < x) {

          fputc(normalized_sorted[input_binary_length - n_index] + (float)log(y/13)*12 + y + dada, tga);
        } else {
          n_index--;
          fputc(normalized_sorted[input_binary_length - n_index] + (float)log(y/13)*12 + y/2.1 + dada+14, tga);
        }
      }else{
        if (x + normalized_input[n_index] > y + normalized_input[n_index - 1]) {
            fputc(normalized_input[n_index] + (float)cos(theta/14.0) - y, tga);
          } else {
            int derp = n_index;
            n_index+=HEIGHT/50;
            fputc(normalized_input[n_index] + 20*(float)y/(float)x + (float)cos(theta/20.0) + 100, tga);
            n_index = derp;
          }
        }

      // RED //

      if (x + normalized_input[n_index] > y + normalized_input[n_index - 1] + 250) {
        n_index--;
        fputc((float)x/((float)y/100) + (float)cos(theta/100.0)*100+x/2.0 + (float)log(y/13)*12  + dada + 15 + normalized_sorted[input_binary_length + n_index], tga);
      } else {
        if ((int)log(x/10.0) % 2 != 0 || x > y) {
          fputc( (float)y/((float)x/100) + (float)cos(log(theta)/100.0)*500+x/2.1 + (float)log(y/53)*12 + log(y/10.0) + dada + normalized_sorted[input_binary_length - n_index], tga);
        } else {
          fputc( (float)y/((float)x/100) + (float)cos(log(theta)/100.0)*500+x/2.1 + (float)log(y/53)*12 + log(y/10.0) + dada + normalized_sorted[input_binary_length + n_index], tga);
        }
      }

      // GREEN //

      if (x + normalized_input[n_index] > (float)log(y/100.0) + normalized_input[n_index - 1]) {
      // if (x > (float)sin(y)) {
        fputc(normalized_input[n_index-HEIGHT-1] + (float)exp(theta/14.0) - y/4.0, tga);
      } else {
        n_index--;
        fputc(normalized_input[n_index-HEIGHT-1] + (float)log(theta/10.0)*y/55.0, tga);
      }
      n_index++;
      theta+=0.001;

      pixel_count++;
      if (n_index > source_size) {
        n_index = 1;
      }
    }
  }

        // ///////////// REWIND ////////////////
      // if ( (ftell(tga) > tga_header_size) && y % 12 == 0) {
      //   tga_current_position = ftell(tga);
      //   fseek(tga, -10L, SEEK_CUR);
      //   pixel_count--;
      // } else {
      //   fseek(tga, 10L, SEEK_CUR);
      //   tga_current_position = ftell(tga);
      // }
      // ///////////// REWIND ////////////////
  // int red_pixel_count = 0;
  // fseek(tga, tga_current_position, SEEK_SET);
  // for (int red_pixel_count = 0; (red_pixel_count <= input_binary_length); red_pixel_count+=9) {
  //   // fseek(tga, 8L, SEEK_CUR); // every red
  //   // fseek(tga, 17L, SEEK_CUR); // every other red
  //   int cur_green = fgetc(tga);
  //   fseek(tga, 26L, SEEK_CUR); // diagonal lines
  //   fputc(cur_green+200, tga);
  //   // fputc(255, tga);
  // }

  // ??
  // int tga_header_size = ftell(tga);
  // int tga_current_position = ftell(tga);;
  // int pixel_count = 0;

  //   for (int x = 0; x < WIDTH && (pixel_count <= input_binary_length); ++x)

  //     if ( (ftell(tga) > tga_header_size) && x > y+50) {
  //       tga_current_position = ftell(tga);
  //       fseek(tga, 1L, SEEK_CUR);
  //       // printf("rewriting\n");
  //     } else {
  //       // printf("___ %d\n", pixel_count);
  //       tga_current_position = ftell(tga);
  //     }

  //     pixel_count++;
  //     if (n_index > source_size) {
  //       n_index = 1;
  //     }




  int green_pixel_count = 0;
  fseek(tga, tga_current_position+2L, SEEK_SET);
  for (int green_pixel_count = 0; (green_pixel_count <= input_binary_length); green_pixel_count+=9) {
    int cur_green = fgetc(tga);
    fseek(tga, 8L, SEEK_CUR); // every red
    // fseek(tga, 17L, SEEK_CUR); // every other red
    // fseek(tga, 26L, SEEK_CUR); // diagonal lines
    fputc(cur_green-200, tga);
  }

  //// magic ends here

  fclose(tga);
  fclose(source);
  printf("^^^^ finished! marvel at your targa!\n");

  return 0;
}
