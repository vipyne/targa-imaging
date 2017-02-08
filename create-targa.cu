// vanessa writes a targa file
// compiling and running this program will produce a targa file

// wip: optionally using CUDA/GPU

// TODO: fix directions
//
// compile : $ gcc create-tga-from-any-input.c -o targa-exe
// usage   : $ ./targa-exe input-file output-filename dimensions
// example : $ ./targa-exe /usr/input/filename.txt /usr/output/filename.tga 1000
//           ==> filename.tga that is 1000 x 1000 pixels large
//
// /Developer/NVIDIA/CUDA-7.5/bin/nvcc targa-cuda.cu -ccbin=$(which clang++-3.7) -o narf && ./narf $INTAR ~/Desktop/output_$(date +%s).tga 1000
// a lot of was originally inspired by Grant Emery's file https://www.tjhsst.edu/~dhyatt/superap/code/targa.c thanks dude
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
  // TODO: add cuda options
  printf("$ ./targa-exe input-file output-filename dimension\n");
}

////// CUDA KERNEL
__global__
void thisIsBasicallyAShaderInMyBook(int n, char *gpu_normalized_input, char *gpu_normalized_sorted, char *gpu_output, int n_index, float theta, int mod)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    gpu_output[i] = gpu_normalized_input[i];
    // i get that i can't do math, because its a char array
    // gpu_output[i] = gpu_normalized_input[i] + mod;
    // but i can't even bitshift? :/
    // gpu_output[i] = gpu_normalized_input[i]>>1;
  }
}

////// MAIN
int main (int argc, char* argv[])
{
  if (argc < 4 || argc > 6) {
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
  int use_cuda = 0; // assume we don't want to use CUDA

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

  if (argv[4] != NULL) {
    use_cuda = 1;
  }

  // intialize and set TARGA header values
  targa_header header;       // variable of targa_header type

  // int x, y;                  // coordinates for `for` loops to pass in
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

  // start to targa write file
  write_header(header, tga);
  printf("^^^^ header written\n");

  // source input file
  fseek(source, 0, SEEK_END);
  int source_size = ftell(source);
  rewind(source);
  printf("^^^^ source file read,      length: %d\n", source_size);

  // width * height = number of pixels
  int final_targa_buffer = WIDTH * HEIGHT * RGBA; // normal people call this a buffer

  // buffer for pixel values (no zeros)
  char normalized_input[final_targa_buffer];

  // buffer for sorted pixel values (still no zeros)
  char normalized_sorted[final_targa_buffer];

  // buffer for entire input file
  char *read_through = (char*) malloc ( sizeof(char) * source_size );

  //////////////////////////////////////
  ///////    //  //  //    ///     /////
  //////  ////  //  // //  //     //////
  /////    //      //    ///  // ///////
  //////////////////////////////////////
  // CUDA buffers
  char *host_buffer;
  host_buffer = (char*)malloc(final_targa_buffer * sizeof(char));
  char *gpu_output;
  char *gpu_normalized_input;
  char *gpu_normalized_sorted;
  int *gpu_n_index;
  float *gpu_theta;

  if (use_cuda) {
    cudaMalloc(&gpu_output, final_targa_buffer * sizeof(char));
    cudaMalloc(&gpu_normalized_input, final_targa_buffer * sizeof(char));
    cudaMalloc(&gpu_normalized_sorted, final_targa_buffer * sizeof(char));
    cudaMalloc(&gpu_n_index, final_targa_buffer * sizeof(int));
    cudaMalloc(&gpu_theta, final_targa_buffer * sizeof(float));
  }
  //////////////////////////////////////
  ///////    //  //  //    ///     /////
  //////  ////  //  // //  //     //////
  /////    //      //    ///  // ///////
  //////////////////////////////////////

  int i = 0;
  int read_through_index = 0;

  fseek(source, SEEK_SET, 0);
  fread(normalized_input, sizeof(char), final_targa_buffer, source); /////////////////////////////////

  // strncpy(normalized_sorted, normalized_input, final_targa_buffer);
  // qsort(normalized_sorted, strlen(normalized_input), sizeof(char), compare_function);

  //////////////////
  //////////////////
  //////////////////
  int wut = 1;
  for (int index_test = 0; index_test < final_targa_buffer; index_test+=3) {
    if (normalized_input[wut] != 0 && normalized_input[wut+1] != 0 && normalized_input[wut+2] != 0 ) {
      normalized_sorted[index_test] = normalized_input[wut];
      normalized_sorted[index_test+1] = normalized_input[wut+1];
      normalized_sorted[index_test+2] = normalized_input[wut+2];
    } else {
      normalized_sorted[index_test] = 160;
      normalized_sorted[index_test+1] = 154;
      normalized_sorted[index_test+2] = 33;
    }
    wut+=3;
  }
  //////////////////
  //////////////////
  //////////////////

  //////////////////////////////////////
  ///////    //  //  //    ///     /////
  //////  ////  //  // //  //     //////
  /////    //      //    ///  // ///////
  //////////////////////////////////////
  if (use_cuda) {
    cudaMemcpy(gpu_output, host_buffer, final_targa_buffer * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_normalized_input, &normalized_sorted, final_targa_buffer * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_normalized_sorted, &normalized_sorted, final_targa_buffer * sizeof(char), cudaMemcpyHostToDevice);
    free(read_through);

    // Magic here / kernel / just a shader
    // kernal_name <<< `execution configuration` >>> (args)
    // <<< grid dimensions (optional), block dimensions / # of thread blocks in grid, # of threads in thread block >>>
    thisIsBasicallyAShaderInMyBook <<< (final_targa_buffer+255)/256, 256 >>>(final_targa_buffer, gpu_normalized_input, gpu_normalized_sorted, gpu_output, 5, 0.06f, 255);

    cudaMemcpy(host_buffer, gpu_output, final_targa_buffer * sizeof(char), cudaMemcpyDeviceToHost);
    fputs(host_buffer, tga); // write to file
    printf("---- used CUDA framework ----\n");
    cudaFree(gpu_output);
    cudaFree(gpu_normalized_input);
    cudaFree(gpu_normalized_sorted);
    free(host_buffer);

  //////////////////////////////////////
  ///////    //  //  //    ///     /////
  //////  ////  //  // //  //     //////
  /////    //      //    ///  // ///////
  //////////////////////////////////////
  } else {
    int n_index = 0;
    float theta = 0;
    for (int y = 0; y < HEIGHT; ++y)
    {
      for (int x = 0; x < WIDTH; ++x)
      {
        fputc( normalized_sorted[n_index], tga);
        fputc( normalized_sorted[n_index+1], tga);
        fputc( normalized_sorted[n_index+2], tga);
        n_index+=3;
      }
    }
    // fputs(normalized_sorted, tga); // write to file
    printf("---- used standard cpu -----\n");
  }

  fclose(tga);
  fclose(source);
  printf("^^^^ finished! marvel at your targa!\n");

  return 0;
}
