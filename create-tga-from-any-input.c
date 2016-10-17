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
	fseek(source, 0L, SEEK_END);
	int source_size = ftell(source);
	printf("asdf asdf \n");

	printf("source size %d\n", source_size);

	rewind(source);

	int input_binary_length = 3 * WIDTH * HEIGHT; // normal people call this a buffer
	
	char normalized_input[input_binary_length];
	

	//char read_through[source_size];;
	char* read_through = malloc(sizeof(char));
	char* read_through_2 = malloc(sizeof(char));

	//read_through[	] = SEEK_END;

	int i = 0;
	int read_through_index = 0;
	int second_index = 0;

	//fflush(source);

	while (i < input_binary_length) 
	{
		if (read_through_index > 1005820)
		{	
			printf("i- %d\n", i);
			printf("inputbinarylen- %d\n", input_binary_length);
			fread(&read_through_2, 1, source_size, source);	
			normalized_input[i] = 1;
			i++;
			//if (read_through_2[second_index] == EOF)
			//	printf("EOF \n");
			//if (read_through_2[second_index] != 0)
			//{	
			//	printf("second-index %d \n", second_index);
			//	normalized_input[i] = read_through_2[second_index];
			//	printf("while if %c, %d\n", normalized_input[i], i);
			//	i++;
			//}
			
		} else {
			
			//printf("while\n");
			fread(&read_through, 1, 1, source);	
			printf("i    %d \n", i);
			//printf("------ %d \n", read_through_index);
			if (read_through[read_through_index] != 0)
			{	
				normalized_input[i] = read_through[read_through_index];
				printf("while if %c, %d\n", normalized_input[i], i);
				i++;
			}
			//printf("after second if %d\n", read_through_index);
		
			read_through_index++;
		}
	}

	//// magic happens here -- write the pixels
	
	for (int y = 0; y < HEIGHT; ++y)
	{
		for (int x = 0; x < WIDTH; ++x)
		{
		fread(&read_through, 1, 1, source);	

		// B G R order
			if (x % 2 == 0)
			//for (int inner = 0; inner < y; ++inner)
			{
				fputc(normalized_input[x]-y, tga);
				fputc(normalized_input[y], tga);
				fputc(normalized_input[x], tga);
			}

			if (x % 2 != 0)
			//for (int inner = 0; inner < x; ++inner) 
			{
				fputc(normalized_input[y+y], tga);
				fputc(normalized_input[y+y], tga);
				fputc(normalized_input[x+y], tga);
			}
		}
	}
	//// magic ends here

	fclose(tga);
	fclose(source);

	return 0;
}
