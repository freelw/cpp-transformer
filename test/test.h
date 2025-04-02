#ifndef TRANSFORMER_TEST_H
#define TRANSFORMER_TEST_H

#include "autograd/node.h"

void test_layernorm();
void test_softmax();
void test_attention(const std::vector<uint> &valid_lens);
void test_attention_without_mask();
void test_attention_with_mask();
void init_qkv_labels1(
    std::vector<autograd::Node *> &queries,
    std::vector<autograd::Node *> &keys,
    std::vector<autograd::Node *> &values,
    std::vector<uint> &labels
);
void init_qkv_labels0(
    std::vector<autograd::Node *> &queries,
    std::vector<autograd::Node *> &keys,
    std::vector<autograd::Node *> &values,
    std::vector<uint> &labels
);
void print_qkv_res_grad(
    const std::vector<autograd::Node *> &queries,
    const std::vector<autograd::Node *> &keys,
    const std::vector<autograd::Node *> &values,
    const std::vector<autograd::Node *> &res
);
void test_mh_attention(
    const std::vector<uint> &valid_lens,
    std::vector<autograd::Node *> queries,
    std::vector<autograd::Node *> keys,
    std::vector<autograd::Node *> values,
    std::vector<uint> labels,
    uint num_hidden,
    uint num_heads = 1);
void test_attention1(const std::vector<uint> &valid_lens);
void test_attention_to_cp_with_mha();
void test_mh_attention_without_mask0();
void test_mh_attention_without_mask1();
void test_mh_attention_without_mask2();
void test_mh_attention_with_mask();
void test_lazy_liner();
void test_pos_encoding();
void test_addnorm();
void test_ffn();
void test_encoder();
void test_mh_attention_with_2d_mask();
void test_decoder();
void test();
#endif