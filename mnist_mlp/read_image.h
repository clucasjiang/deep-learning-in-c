#ifndef READ_IMAGE_H
#define READ_IMAGE_H

void read_image(float img[784], int train, int img_index);
int read_label(int train, int img_index);
void print_img(float img[784]);

#endif
