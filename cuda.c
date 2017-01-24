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
#define PI 3.14159265358979323846

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


////// CUDA KERNEL
__global__
void thisIsBasicallyAShaderInMyBook(int n, float a, float *x, float *y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];

	//printf("^^^^ writing pixels \n");
  int n_index = 0;
	float theta = 0.0;

  //// magic happens here
  for (int y = 0; y < HEIGHT; ++y)
  {
    for (int x = 0; x < WIDTH; ++x)
    {
      // pixels read in B G R order
      fputc( fabsf((x%200)-(y)/(float)sin(x-y) ), tga); ////// BLUE

			float butter = sin(theta*10.0)* (250.0-y) - (float)sin(theta/250.0)*200.0 - 200.0;

			if (y > butter+(x-50)*(float)log(100) ) { ////// IF
				n_index++;
				fputc(normalized_input[n_index] + (float)sin(n_index*theta)-y/5, tga); ///// GREEN
				fputc( normalized_sorted[n_index]+(float)log(normalized_sorted[n_index]*500)-x/40, tga); ///// RED
			} else {
				n_index--;
				fputc( (normalized_sorted[n_index]) + ((x-y)/(float)brain), tga); ////// GREEN
				n_index--;
				fputc( (normalized_input[n_index] - (float)sin(theta/10.0) - x - y/2) / 3.7 + butter/3, tga); ///// RED
			}			
      n_index++;
			theta+=0.001;
    }
  }
  //// magic ends here
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

	// CUDA buffers for the above
	int N = source_size;
	int *gpu_input_binary_length;
	char *gpu_normalized_input;
	char *gpu_normalized_input;
	char *gpu_read_through;

	cudaMalloc(&gpu_input_binary_length, N * sizeof(int));
	cudaMalloc(&gpu_normalized_input, N * sizeof(char));
	cudaMalloc(&gpu_normalized_sorted, N * sizeof(char));
	cudaMalloc(&gpu_read_through, N * sizeof(char));
	
  int i = 0;
  int read_through_index = 0;
  while (i < input_binary_length)
  {
    fread(read_through, 1, source_size, source);
    if (read_through_index >= source_size)
    {
			//printf("rewinding\n");
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
	
	cudaMemcpy(gpu_input_binary_length, input_binary_length, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_normalized_input, normalized_input, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_normalized_sorted, normalized_input, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_read_through, read_through_index, N * sizeof(int), cudaMemcpyHostToDevice);

	// Magic here / kernel / just a shader

	// kernal_name <<< `execution configuration` >>> (args)
	// kernal_name <<< grid dimensions (optional), block dimensions, # of threads >>> (args)
	thisIsBasicallyAShaderInMyBook <<< A, B >>>( );

	cudaMemcpy(input_binary_length, gpu_input_binary_length, N * sizeof(int), cudaMemcpyDeviceToHost);
  
	cudafree(gpu_input_binary_length);
	cudafree(gpu_normalized_input);
	cudafree(gpu_normalized_sorted);
	cudafree(gpu_read_through);

	free(input_binary_length);
	free(normalized_input);
	free(normalized_sorted);
	free(read_through);
	
	fclose(tga);
  fclose(source);
  printf("^^^^ finished! marvel at your targa!\n");

  return 0;
}
