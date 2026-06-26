#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

FILE *test_imgs_ptr;
FILE *test_labels_ptr;
int img_index = 0;

uint32_t convert_msb(uint32_t msb_int) {
    uint32_t lsb_int = ((msb_int & 0xFF000000) >> 24) + ((msb_int & 0x00FF0000) >> 8) + ((msb_int & 0x0000FF00) << 8) + ((msb_int & 0x000000FF) << 24);
    return lsb_int;
}

void read_image(float img[784], int index) { // img is the array this function will write to, index is the index of the image in the test dataset
    test_imgs_ptr = fopen("data/MNIST/raw/t10k-images-idx3-ubyte", "rb");
    uint8_t img_byte[784];
    fseek(test_imgs_ptr, 16 + (index * 784), SEEK_SET);
    fread(img_byte, 1, 784, test_imgs_ptr);
    for (int i=0; i<784; i++) {
        img[i] = img_byte[i] / 255.0f;
    }
}

int read_label(int index) {
    test_labels_ptr = fopen("data/MNIST/raw/t10k-labels-idx1-ubyte", "rb");
    fseek(test_labels_ptr, 8+index, SEEK_SET);
    uint8_t label;
    fread(&label, 1, 1, test_labels_ptr);
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

int main() {
    float img[784];
    read_image(img, img_index);
    return 0;
}
