#ifndef TRANSFORMER_MACRO_H
#define TRANSFORMER_MACRO_H

// #define DEBUG_GRAD
#define MAX_POSENCODING_LEN 10000
#define RESOURCE_NAME "../../resources/fra_preprocessed.txt"
#define SRC_VOCAB_NAME "../fra_vocab_builder/vocab_en.txt"
#define TGT_VOCAB_NAME "../fra_vocab_builder/vocab_fr.txt"
#define SRC_VOCAB_TINY_NAME "../fra_vocab_builder/vocab_en_tiny.txt"
#define TGT_VOCAB_TINY_NAME "../fra_vocab_builder/vocab_fr_tiny.txt"
#define TEST_FILE "./test.txt"
#define BATCH_SIZE 128
#define NUM_STEPS 9
#define NUM_HIDDENS 256
#define FFN_NUM_HIDDENS 64
#define TINY_NUM_HIDDENS 16
#define TINY_FFN_NUM_HIDDENS 4
#define NUM_HEADS 4
#define NUM_BLKS 2

#endif