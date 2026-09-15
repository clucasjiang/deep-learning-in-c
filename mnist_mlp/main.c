#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlp.h"
#include "mlp_weights.h"
#include "read_image.h"

void run_model(int img_index) {
    if (img_index >= 10000 || img_index < 0) {
        printf("img_index is an index between 0-9999\n");
        return;
    }
    float input[784]; 
    ForwardCache cache = {0};
    load_stored_weights();
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

void evaluate_model(void) {
    float input[784];
    ForwardCache cache;
    int correct = 0;
    float total_loss = 0;
    load_stored_weights();

    for (int index=0; index<10000; index++) {
        read_image(input, 0, index);
        int label = read_label(0, index);
        forward(input, &cache);
        int prediction = 0;
        for (int i=1; i<10; i++) {
            if (cache.probs[i] > cache.probs[prediction]) {
                prediction = i;
            }
        }
        if (prediction == label) {
            correct++;
        }
        total_loss += find_cross_entropy_cost(cache.probs, label);
    }
    printf("Test accuracy: %.2f%% (%d/10000)\n", 100.0f * correct / 10000, correct);
    printf("Test loss: %.4f\n", total_loss / 10000);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <image_index|train|eval>\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "train") == 0) {
        printf("Starting training...\n");
        return train_model();
    } else if (strcmp(argv[1], "eval") == 0) {
        evaluate_model();
    } else {
        char *end;
        long img_index = strtol(argv[1], &end, 10);
        if (end == argv[1] || *end != '\0' || img_index < 0 || img_index >= 10000) {
            fprintf(stderr, "Use train, eval, or an image index between 0-9999.\n");
            return 1;
        }
        run_model((int) img_index);
    }
    return 0;
}
