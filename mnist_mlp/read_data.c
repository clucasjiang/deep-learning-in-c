#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "read_data.h"

#define TEST_IMAGES_FILE "t10k-images-idx3-ubyte"
#define TEST_LABELS_FILE "t10k-labels-idx1-ubyte"

static FILE *open_mnist_file(const char *filename) {
    const char *roots[] = {
        "mnist_dataset/MNIST/raw/",
        "../mnist_dataset/MNIST/raw/",
    };
    char path[256];

    for (size_t i=0; i<sizeof(roots) / sizeof(roots[0]); i++) {
        snprintf(path, sizeof(path), "%s%s", roots[i], filename);
        FILE *file = fopen(path, "rb");
        if (file != NULL) {
            return file;
        }
    }

    fprintf(stderr, "Could not open MNIST file: %s\n", filename);
    exit(1);
}

// uint32_t convert_msb(uint32_t msb_int) {
//     uint32_t lsb_int = ((msb_int & 0xFF000000) >> 24) + ((msb_int & 0x00FF0000) >> 8) + ((msb_int & 0x0000FF00) << 8) + ((msb_int & 0x000000FF) << 24);
//     return lsb_int;
// }

void read_image(float img[784], int img_index) { // img is the array this function will write to, index is the index of the image in the test dataset
    FILE *test_imgs_ptr = open_mnist_file(TEST_IMAGES_FILE);
    uint8_t img_byte[784];
    fseek(test_imgs_ptr, 16 + (img_index * 784), SEEK_SET);
    fread(img_byte, 1, 784, test_imgs_ptr);
    for (int i=0; i<784; i++) {
        img[i] = img_byte[i] / 255.0f;
    }
    fclose(test_imgs_ptr);
}

int read_label(int img_index) {
    FILE *test_labels_ptr = open_mnist_file(TEST_LABELS_FILE);
    fseek(test_labels_ptr, 8+img_index, SEEK_SET);
    uint8_t label;
    fread(&label, 1, 1, test_labels_ptr);
    fclose(test_labels_ptr);
    return label;
}

void print_img(float img[784]){
    for (int i=0; i<28; i++) {
        for (int j=0; j<28; j++) {
            printf("%.1f ", img[i*28+j]);
        }
        printf("\n");
    }
}
