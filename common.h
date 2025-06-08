#ifndef COMMON_H
#define COMMON_H


#include "backends/cpu/cpu_ops.h"
#include "backends/gpu/cuda_ops.h"
#include "graph/node.h"

extern BackendOps* g_backend_ops;
extern bool g_training;
void zero_grad();
void zero_c_tensors();
void print_no_zero_tensor_names();
void insert_boundary_action();
void init_backend();
void release_backend();
void construct_env();
void destruct_env();
void use_gpu(bool use = true);
bool is_use_gpu();
void print_all_tensors();

#define NUM_STEPS 9 // 对齐 dl2 dataloader
#define MAX_POSENCODING_LEN 1000 // 对齐 dl2
#define RESOURCE_NAME "./resources/fra_preprocessed_512.txt"
#define SRC_VOCAB_NAME "./vocab/fra_vocab_builder/vocab_en.txt"
#define TGT_VOCAB_NAME "./vocab/fra_vocab_builder/vocab_fr.txt"
#define TEST_FILE "./test.txt"
#define TEST_LM_FILE "./test_lm.txt"
#define LM_NUM_STEPS 5
#define TIMEMACHINE_VOCAB_NAME "./vocab/time_machine/vocab.txt"
#define TIMEMACHINE_RESOURCE_NAME "./resources/time_machine/timemachine_preprocessed.txt"

#endif