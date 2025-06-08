#include "common.h"
#include "checkpoint.h"
#include "dataloader.h"
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

    use_gpu(gpu == 1);
    construct_env();
    zero_c_tensors();
    zero_grad();

    LMDecoder* lm_decoder = new LMDecoder(
        dec_vocab_size, num_hiddens, ffn_num_hiddens,
        num_heads, num_blks, max_posencoding_len, dropout
    );

    delete lm_decoder;
    destruct_env();
    return 0;
}