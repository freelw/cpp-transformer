#include "common.h"
#include "checkpoint.h"
#include "dataloaders/translation/dataloader.h"
#include "module/language_model/lm_decoder.h"
#include "optimizers/adam.h"
#include <unistd.h>
#include <signal.h>

extern bool shutdown;
void signal_callback_handler(int signum);

void check_parameters(const std::vector<Parameter*>& parameters, int num_blks) {
    int parameters_size_should_be = 0;
    parameters_size_should_be += 1; // target embedding
    parameters_size_should_be += num_blks * (
        1 + // decoder block attention1 wq
        1 + // decoder block attention1 wk
        1 + // decoder block attention1 wv
        1 + // decoder block attention1 wo
        1 + // decoder block addnorm1 gamma
        1 + // decoder block addnorm1 beta
        1 + // decoder block ffn w1
        1 + // decoder block ffn b1
        1 + // decoder block ffn w2
        1 + // decoder block ffn b2
        1 + // decoder block addnorm2 gamma
        1 // decoder block addnorm2 beta
        );
    parameters_size_should_be += 1; // target linear w
    parameters_size_should_be += 1; // target linear b
    assert(parameters.size() == parameters_size_should_be);
    assert(parameters_size_should_be == 27);
}

void print_progress(const std::string& prefix, uint i, uint tot) {
    std::cout << "\r" << prefix << " [" << i << "/" << tot << "]" << std::flush;
}

std::vector<uint> trim_or_padding(const std::vector<uint>& src, uint max_len, uint pad_id) {
    std::vector<uint> res = src;
    if (src.size() > max_len) {
        res.resize(max_len);
    } else {
        res.resize(max_len, pad_id);
    }
    return res;
}

void load_tokens_from_file(
    seq2seq::DataLoader& loader,
    std::vector<std::vector<uint>>& src_token_ids,
    std::vector<std::vector<uint>>& tgt_token_ids,
    int& enc_vocab_size,
    int& dec_vocab_size,
    int& bos_id,
    int& eos_id,
    int& src_pad_id,
    int& tgt_pad_id
) {
    loader.get_token_ids(src_token_ids, tgt_token_ids);
    enc_vocab_size = loader.src_vocab_size();
    dec_vocab_size = loader.tgt_vocab_size();
    bos_id = loader.tgt_bos_id();
    eos_id = loader.tgt_eos_id();
    src_pad_id = loader.src_pad_id();
    tgt_pad_id = loader.tgt_pad_id();
}

void init_dec_valid_lens(Tensor* dec_valid_lens) {
    int32_t* dec_valid_lens_buffer = static_cast<int32_t*>(::malloc(
        dec_valid_lens->size()
    ));

    auto shape = dec_valid_lens->get_shape();

    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            dec_valid_lens_buffer[i * shape[1] + j] = j + 1;
        }
    }

    g_backend_ops->cp_to_device(
        dec_valid_lens,
        reinterpret_cast<char*>(dec_valid_lens_buffer),
        dec_valid_lens->size()
    );

    ::free(dec_valid_lens_buffer);
}

int main(int argc, char* argv[]) {

    shutdown = false;

    int opt;
    int epochs = 10;
    int batch_size = 128;
    int gpu = 1;
    float lr = 0.001f;
    std::string checkpoint;
    std::string corpus = TIMEMACHINE_RESOURCE_NAME;

    while ((opt = getopt(argc, argv, "f:c:e:l:b:g:")) != -1) {
        switch (opt) {
        case 'f':
            corpus = optarg;
            break;
        case 'c':
            checkpoint = optarg;
            break;
        case 'e':
            epochs = atoi(optarg);
            break;
        case 'l':
            lr = atof(optarg);
            break;
        case 'b':
            batch_size = atoi(optarg);
            break;
        case 'g':
            gpu = atoi(optarg);
            break;
        default:
            std::cerr << "Usage: " << argv[0]
                << " -f <corpus> -c <checpoint> -e <epochs>" << std::endl;
            return 1;
        }
    }

    std::cout << "corpus : " << corpus << std::endl;
    std::cout << "epochs : " << epochs << std::endl;
    std::cout << "batch_size : " << batch_size << std::endl;
    std::cout << "gpu : " << gpu << std::endl;
    std::cout << "learning rate : " << lr << std::endl;
    std::cout << "checkpoint : " << checkpoint << std::endl;

    int num_hiddens = 256;
    int num_blks = 2;
    float dropout = 0.2f;
    int ffn_num_hiddens = 64;
    int num_heads = 4;
    int num_steps = NUM_STEPS;
    int max_posencoding_len = MAX_POSENCODING_LEN;

    std::string src_vocab_name = SRC_VOCAB_NAME;
    std::string tgt_vocab_name = TIMEMACHINE_VOCAB_NAME;
    std::string test_file = TEST_FILE;
    seq2seq::DataLoader loader(corpus, src_vocab_name, tgt_vocab_name, test_file);

    int enc_vocab_size = 0;
    int dec_vocab_size = 0;
    int bos_id = 0;
    int eos_id = 0;
    int src_pad_id = 0;
    int tgt_pad_id = 0;

    std::vector<std::vector<uint>> v_src_token_ids;
    std::vector<std::vector<uint>> v_tgt_token_ids;
    load_tokens_from_file(
        loader,
        v_src_token_ids, v_tgt_token_ids,
        enc_vocab_size, dec_vocab_size,
        bos_id,
        eos_id,
        src_pad_id,
        tgt_pad_id
    );

    bool predicting = epochs == 0;
    g_training = !predicting;
    if (predicting) {
        batch_size = 1; // set batch size to 1 for predicting
    }

    use_gpu(gpu == 1);
    construct_env();
    zero_c_tensors();
    zero_grad();

    LMDecoder* lm_decoder = new LMDecoder(
        dec_vocab_size, num_hiddens, ffn_num_hiddens,
        num_heads, num_blks, max_posencoding_len, dropout
    );

    Tensor* tgt_token_ids = allocTensor({ batch_size, num_steps }, INT32);
    Tensor* dec_valid_lens = predicting ? allocTensor({ 1 }, INT32) : allocTensor({ batch_size, num_steps }, INT32);
    Tensor* labels = allocTensor({ batch_size * num_steps }, INT32);
    Tensor* ce_mask = allocTensor({ batch_size * num_steps });

    int32_t* tgt_token_ids_buffer = static_cast<int32_t*>(::malloc(
        tgt_token_ids->size()
    ));
    int32_t* labels_buffer = static_cast<int32_t*>(::malloc(
        labels->size()
    ));
    float* ce_mask_buffer = static_cast<float*>(::malloc(
        ce_mask->size()
    ));

    auto res = lm_decoder->forward(tgt_token_ids, dec_valid_lens);
    auto loss = res->reshape({ -1, dec_vocab_size })->CrossEntropy(labels)->mask(ce_mask)->avg_1d(ce_mask);
    insert_boundary_action();

    std::vector<Parameter*> parameters = lm_decoder->get_parameters();
    check_parameters(parameters, num_blks);
    Adam adam(parameters, lr);
    loss->backward();
    adam.clip_grad(1.0f);
    adam.step();
    graph::validateAllNodesRefCnt(0);
    // printAllActions();
    allocMemAndInitTensors();
    std::cout << "Allocating memory  " << std::endl
        << "for tensors : " << tensors_data_capacity << " bytes, " << std::endl
        << "for c_tensors: " << c_tensors_data_capacity << " bytes " << std::endl
        << "for grad_tensors: " << grad_tensors_data_capacity << " bytes" << std::endl;
    gDoOnceActions();

    if (!checkpoint.empty()) {
        std::cout << "loading from checkpoint : " << checkpoint << std::endl;
        disableInitWeightAction();
        loadfrom_checkpoint(checkpoint, parameters);
        std::cout << "loaded from checkpoint" << std::endl;
    }

    if (predicting) {
    } else {
        init_dec_valid_lens(dec_valid_lens);
        signal(SIGINT, signal_callback_handler);
        int epoch = 0;
        for (; epoch < epochs; ++epoch) {

        }


    }
    ::free(tgt_token_ids_buffer);
    ::free(labels_buffer);
    ::free(ce_mask_buffer);
    delete lm_decoder;
    destruct_env();
    return 0;
}