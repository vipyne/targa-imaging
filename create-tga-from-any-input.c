// vanessa writes a targa file
// compiling and running this file will produce a targa file
// a lot of this is based on Grant Emery's file https://www.tjhsst.edu/~dhyatt/superap/code/targa.c thanks dude
// author: vanessa pyne --- github.com/vipyne


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

#define BYTE_RANGE 256
//#define WIDTH 250
//#define HEIGHT 250

////// targa file header

typedef struct {
	char id_length;      // length of id field (number of bytes - max 255)
	char map_type;       // colormap field (0 or 1; no map or 256 entry palette)
	char image_type;     // ( 0 - no image data included
						 //   1 - uncompressed, color mapped image
						 //	  2 - uncompressed, RGB image
						 //	  3 - uncompressed, black & white image
						 //	  9 - run-length encoded(RLE-lossless compression),color mapped image
						 //	 10 - RLE, RGB image
						 //	 11 - compressed, black & white image )

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

//int filter_zeros (char* input, char* normalized_input)
//{
	//char* input_arr = malloc(size_of(int));
	//*input_arr = *input[
	//int normalized_input_length = length; 
//}

int little_endianify (int number)
{
	return number % BYTE_RANGE;
}

int big_endianify (int number)
{
	return number / BYTE_RANGE;
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

////// MAIN

int main (int argc, char* argv[])
{
	FILE *tga;                 // pointer to file that we will write
	targa_header header;       // variable of targa_header type

	int x, y;                  // coordinates for `for` loops to pass in
		    		 		   // correct number of values

	int HEIGHT = atoi(argv[3]);
	int WIDTH = atoi(argv[3]);
	// set header values

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
	header.misc = 0x20;       // scan from upper left corner, wut dude

	// start to write file

	tga = fopen(argv[2], "wb");

	write_header(header, tga);

	FILE *source;

	source = fopen(argv[1], "rb");

	int input_binary_length = 3 * WIDTH * HEIGHT; // normal people call this a buffer
	
	//char input[input_binary_length];
	char normalized_input[input_binary_length];
	
	char* read_through = malloc(sizeof(char));
	read_through[0] = 
	fread(&read_through, 1, input_binary_length, source);	
	
	//int x = 0;
	//in//t i = 0;

	//while (normalized_input[i] != '\0')
	//{
	//	if (input[x] != 0)
	//	{
	//		normalized_input[i] = input[x];
	//		++i;
	//	}
	//	++x;
	//}
	//return normalized_inpu;
	//fseek(

	//circular buffer?

	
	//// magic happens here -- write the pixels
	
	//filter_zeros( &input, &normalized_input);  


	for (int y = 0; y < HEIGHT; ++y)
	{
		for (int absolute_x = 0; absolute_x < WIDTH; ++absolute_x)
		{
		int x = absolute_x;
		fread(&read_through, 1, 1, source);	
		//bool flag = true;
		//	if (input == EOF) 
		//		rewind(source);
		
		//if (input[x] == 0)
		//{
			//x += 1;
			//flag = false;
			//fputc(170, tga);
			//fputc(100, tga);
			//fputc(30, tga);

			//WIDTH += 1;
		//}
		//x = inefficient_ignore_zeros(absolute_x, &input_binary_length, input);

		//if (flag)
		//{
		// B G R order
			if (x % 2 == 0)
			//for (int inner = 0; inner < y; ++inner)
			{
				
				//printf("asf\n");
				fputc(read_through[x]-y, tga);
				fputc(read_through[y], tga);
				fputc(read_through[x], tga);
			}

			if (x % 2 != 0)
			//for (int inner = 0; inner < x; ++inner) 
			{
				//printf("xxxx\n");
				fputc(read_through[y+y], tga);
				fputc(read_through[y+y], tga);
				fputc(read_through[x+y], tga);
			}
			//if ( x % 2 == 0 ) 
			//{
			//	fputc(input[x * y] + x, tga);
			//	fputc(input[x * x] + x, tga);
			//	fputc(input[y * y] + y, tga);
			// } else {
			//	fputc(input[x + y] + x, tga);
			//	fputc(input[y], tga);
			//	fputc(input[x + y], tga);
			//}
		}
	}
	//// magic ends here

	fclose(tga);
	fclose(source);

	return 0;
}
