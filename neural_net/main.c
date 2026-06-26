#include <stdio.h>
#include <stdlib.h>

#include "neural_net.h"
#include "read_data.h"

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Specify an image index from the MNIST dataset. Usage: %s <image_index>\n", argv[0]);
        return 1;
    }

    int img_index = atoi(argv[1]);
    if (img_index >= 10000 || img_index < 0) {
        printf("img_index is an index between 0-9999\n");
        return 1;
    }

    float input[784]; float output[10];
    read_image(input, img_index);
    run_model(784, 10, input, output);
    float max_prob = 0;
    int max_index = 0;
    for (int i=0; i<10; i++) {
        if (output[i] > max_prob) {
            max_prob = output[i];
            max_index = i;
        }
    }
    printf("Predicted digit: %d  (Confidence: %.2f%%)\n", max_index, max_prob * 100.0);
    printf("Actual digit: %d\n", read_label(img_index));

    return 0;
}
