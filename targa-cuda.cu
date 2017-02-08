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
void thisIsBasicallyAShaderInMyBook(int n, char *gpu_normalized_input, char *gpu_normalized_sorted, char *gpu_output, int n_index, float theta)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // int n_index = 0;
    // float theta = 0;
    gpu_output[i] = gpu_normalized_input[i];
    // gpu_output[i] = gpu_normalized_input[n_index[i]-1000-1] + (float)log(theta[i]/10.0);

    // for (int y = 0; y < 1000; ++y)
    // {
    //   for (int x = 0; x < 1000; ++x)
    //   {
    //     int line = (float)cos(theta/10.0)*1 + 10*x;
    //     float dada = sin(theta) * 100.0;
    //     // BLUE //
    //     if (x % 2 ==0) {
    //       if (y > line+ gpu_normalized_input[n_index - 1] || y < x) {
    //         gpu_output[i] = gpu_normalized_sorted[n - n_index] + (float)log(y/13)*12 + y + dada;
    //       } else {
    //         n_index--;
    //         gpu_output[i] = gpu_normalized_sorted[n - n_index] + (float)log(y/13)*12 + y + dada+25;
    //       }
    //     }else{
    //       if (x + gpu_normalized_input[n_index] > y + gpu_normalized_input[n_index - 1]) {
    //           gpu_output[i] = gpu_normalized_input[n_index] + (float)cos(theta/14.0) - y;
    //       } else {
    //         int derp = n_index;
    //         n_index+=1000/50;
    //         gpu_output[i] = gpu_normalized_input[n_index] + 20*(float)y/(float)x + (float)cos(theta/20.0);
    //         n_index = derp;
    //       }
    //     }
    //     // GREEN //
    //     if (x + gpu_normalized_input[n_index] > y + gpu_normalized_input[n_index - 1]) {
    //       gpu_output[i] = gpu_normalized_input[n_index-1000-1] + (float)exp(theta/14.0) - y/4.0;
    //     } else {
    //       gpu_output[i] = gpu_normalized_input[n_index-1000-1] + (float)log(theta/10.0)*y/55.0;
    //     }
    //     // RED //
    //     if (x + gpu_normalized_input[n_index] > y + gpu_normalized_input[n_index - 1] + 250) {
    //       n_index--;
    //       gpu_output[i] = gpu_normalized_sorted[n + n_index] - (float)x/((float)y/100) + (float)cos(theta/100.0)*125+x/2.0 + (float)log(y/13)*12 + y + dada + 25;
    //     } else {
    //       n_index--;
    //       gpu_output[i] = gpu_normalized_sorted[n - n_index] - (float)y/((float)x/100) + (float)cos(theta/100.0)*525+x/2.0 + (float)log(y/13)*12 + y + dada;
    //     }
        // n_index++;
        // theta+=0.001;
    //   }
    // }
    // gpu_output[i] = gpu_normalized_input[i];
  }
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

	//char *normalized_input;
	//normalized_input = (char*)malloc(input_binary_length * sizeof(char));

	char *host_buffer;
  host_buffer = (char*)malloc(input_binary_length * sizeof(char));

  // buffer for sorted pixel values (still no zeros)
  char normalized_sorted[input_binary_length];

  // buffer for entire input file
  char *read_through = (char*) malloc ( sizeof(char) * source_size );


 // CUDA buffers
  int N = input_binary_length;
  char *gpu_output;
  char *gpu_normalized_input;
  char *gpu_normalized_sorted;
  int *gpu_n_index;
  float *gpu_theta;

  cudaMalloc(&gpu_output, N * sizeof(char));
  cudaMalloc(&gpu_normalized_input, N * sizeof(char));
  //cudaMalloc(&gpu_normalized_sorted, N * sizeof(char));
  cudaMalloc(&gpu_normalized_sorted,  N * sizeof(char));
  cudaMalloc(&gpu_n_index,  N * sizeof(int));
  cudaMalloc(&gpu_theta,  N * sizeof(float));
  printf("^^^^ !! d      N                       : %d \n", N);
  printf("^^^^ !! d                  sizeof(char): %d \n", (int)sizeof(char));
  printf("^^^^ !! lu                 sizeof(char): %lu \n", sizeof(char));
  printf("^^^^ !! lu     sizeof(normalized_input): %lu \n", sizeof(normalized_input));
  printf("^^^^ !! lu sizeof(gpu_normalized_input): %lu \n", sizeof(*gpu_normalized_input));

  printf("N * sizeof(char) : %lu\n", N * sizeof(char) );
  //printf("input_binary_length / N : %d / %d\n", input_binary_length, N);







  int i = 0;
  int read_through_index = 0;


// fread(buffer, strlen(c)+1, 1, fp);


  fseek(source, SEEK_SET, 0);
  // fread(read_through, 1, source_size, source);
  fread(normalized_input, sizeof(char), N, source); /////////////////////////////////
  // fread(&normalized_input, strlen(normalized_input), 1, source);


  printf("read_through : %s\n", normalized_input);
  printf("normalized_input : %lu\n", sizeof(normalized_input));
  printf("normalized_input : %lu\n", strlen(normalized_input));
  printf("normalized_input[0] : %c\n", normalized_input[0]);
  printf("normalized_input[421] : %c\n", normalized_input[421]);
  // strncpy(normalized_input, read_through, sizeof(normalized_input));
  // strncpy(normalized_input, read_through, input_binary_length);
  // while (i < input_binary_length)
  // {
  //   normalized_input[i] = read_through[read_through_index];

  //   // fread(read_through, 1, source_size, source);
  //   // if (read_through_index >= source_size)
  //   // {
  //   //   //printf("rewinding\n");
  //   //   rewind(source);
  //   //   read_through_index = 0;
  //   // }
  //   // if (read_through[read_through_index] != '0')
  //   // {
  //   //   normalized_input[i] = read_through[read_through_index];
  //     i++;
  //   // }
  //   read_through_index++;
  // }
  printf("^^^^ normalized buffer set, length: %d \n", (int) sizeof(normalized_input));
  printf("^^^^ read_through buffer set, length: %d \n", (int) sizeof(read_through));
  printf("^^^^ normalized buffer set, length: %d \n", (int) sizeof(normalized_input));

  // strncpy(normalized_sorted, normalized_input, input_binary_length);
  // qsort(normalized_sorted, strlen(normalized_input), sizeof(char), compare_function);





//////////////////
//////////////////
//////////////////

  int wut = 1;
  for (int index_test = 0; index_test < input_binary_length; index_test+=3) {
    if (normalized_input[wut] != 0) {
      normalized_sorted[index_test] = normalized_input[wut];
      normalized_sorted[index_test+1] = normalized_input[wut];
      // normalized_sorted[index_test+1] = abs(normalized_input[wut]);
      normalized_sorted[index_test+2] = normalized_input[wut];
    } else {
      normalized_sorted[index_test] = 120;
      normalized_sorted[index_test+1] = 4;
      normalized_sorted[index_test+2] = 33;
    }
    wut++;
    // printf("wut: %d\n", wut);
    // // printf("normalized_input[4]: %d\n", normalized_input[4]);
    // // printf("normalized_sorted[index_test]: %d\n", normalized_sorted[index_test]);
    // // printf("normalized_input[73]: %d\n", normalized_input[73]);
    // printf("normalized_input[wut]: %d\n", normalized_input[wut]);
  }

//////////////////
//////////////////
//////////////////

printf("cuda mem copy\n");
  // for (int i = 0; i < input_binary_length-1; i++) {
  //   normalized_sorted[i] = normalized_input[i];
  // }
  cudaMemcpy(gpu_output, host_buffer, N * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_normalized_input, &normalized_sorted, N * sizeof(char), cudaMemcpyHostToDevice);
  // cudaMemcpy(gpu_normalized_input, &normalized_input, N * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_normalized_sorted, &normalized_sorted, N * sizeof(char), cudaMemcpyHostToDevice);
  //cudaMemcpy(gpu_normalized_sorted, &normalized_sorted, N * sizeof(char), cudaMemcpyHostToDevice);
  //printf("^^^^ CUDA input buffer set, length: %lu \n", sizeof(gpu_normalized_input));
  //printf("^^^^ CUDA sorted buffer set, length: %lu \n", sizeof(gpu_normalized_sorted));
  //printf("^^^^ CUDA output buffer set, length: %lu \n", sizeof(gpu_output));
  free(read_through);

//   int *n_index = (int*)malloc(sizeof(int) * N);
//   float *theta = (float*)malloc(sizeof(float) * N);
//   //n_index[0] = 0;
//   //theta[0] = 0.001;
//   for (int n_i = 0; n_i < N; ++n_i) {
//     n_index[n_i] = n_i;
//     theta[n_i] = (n_i + 1)/1000.0;
//   }
// printf("1   guess here\n");
//   // cudaMemcpy(gpu_output, host_buffer, N * sizeof(char), cudaMemcpyHostToDevice);
//   cudaMemcpy(gpu_n_index, &n_index, N * sizeof(int), cudaMemcpyHostToDevice);
// printf("2 guess here\n");
//   cudaMemcpy(gpu_theta, &theta, N * sizeof(float), cudaMemcpyHostToDevice);

printf("guess here\n");

  // Magic here / kernel / just a shader ////////////////////////
  // kernal_name <<< `execution configuration` >>> (args)
  // <<< grid dimensions (optional), block dimensions / # of thread blocks in grid, # of threads in thread block >>>
  thisIsBasicallyAShaderInMyBook <<< (N+255)/256, 256 >>>(N, gpu_normalized_input, gpu_normalized_sorted, gpu_output, 5, 0.06);
  // thisIsBasicallyAShaderInMyBook <<< (N+255)/256, 256 >>>(N, gpu_normalized_input, gpu_normalized_sorted, gpu_output, n_index, theta);
  ////////////// <<< >>> //////////////////

printf("meh\n");

	cudaMemcpy(host_buffer, gpu_output, N * sizeof(char), cudaMemcpyDeviceToHost);

	printf("size of host buffer : %lu\n", sizeof(host_buffer));

  fputs(host_buffer, tga); //////////////////////////
   // int n_index = 0;
   // for (int y = 0; y < HEIGHT; ++y)
   // {
   //   for (int x = 0; x < WIDTH; ++x)
   //  {
			// fputc(normalized_input[n_index],tga);
			// fputc(normalized_input[n_index],tga);
			// fputc(normalized_input[n_index],tga);
   //     n_index++;
   //   }
   // }

	printf("\n");

  cudaFree(gpu_output);
  cudaFree(gpu_normalized_input);
  cudaFree(gpu_normalized_sorted);

  free(host_buffer);
	//free(normalized_input);

	fclose(tga);
  fclose(source);
  printf("^^^^ finished! marvel at your targa!\n");

  return 0;
}
