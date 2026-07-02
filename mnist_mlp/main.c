#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlp.h"
#include "read_data.h"

void run_model(int img_index) {
    if (img_index >= 10000 || img_index < 0) {
        printf("img_index is an index between 0-9999\n");
        return;
    }
    float input[784]; 
    ForwardCache cache = {0};
    read_image(input, 0, img_index);
    forward(input, &cache);
    float max_prob = 0;
    int max_index = 0;
    for (int i=0; i<10; i++) {
        if (cache.probs[i] > max_prob) {
            max_prob = cache.probs[i];
            max_index = i;
        }
    }
    printf("Predicted digit: %d  (Confidence: %.2f%%)\n", max_index, max_prob * 100.0);
    printf("Actual digit: %d\n", read_label(0, img_index));
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Specify an image index from the MNIST dataset, or train. Usage: %s <image_index|train>\n", argv[0]);
        return 1;
    }

    int img_index = atoi(argv[1]);
    if (strcmp(argv[1], "train") == 0) {
        printf("Starting training...\n");
        train_model();
    } else {
        run_model(img_index);
    }
    return 0;
}
