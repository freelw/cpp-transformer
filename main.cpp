#include <iostream>
#include <signal.h>
#include <unistd.h>
#include "layernorm.h"
#include "attention.h"
#include "posencoding.h"
#include "addnorm.h"
#include "ffn.h"
#include "test.h"
#include "dataloader.h"
#include "autograd/optimizers.h"
#include "seq2seq.h"
#include "checkpoint.h"
#include "macro.h"

using namespace std;

bool shutdown = false;
void signal_callback_handler(int signum);

void print_progress(uint i, uint tot) {
    std::cout << "\r[" << i << "/" << tot << "]" << std::flush;
}

std::vector<uint> trim_or_padding(const std::vector<uint> &src, uint max_len, uint pad_id) {
    std::vector<uint> res = src;
    if (src.size() > max_len) {
        res.resize(max_len);
    } else {
        res.resize(max_len, pad_id);
    }
    return res;
}

std::vector<uint> add_bos(const std::vector<uint> &src, uint bos_id) {
    std::vector<uint> res = src;
    res.insert(res.begin(), bos_id);
    return res;
}

Seq2SeqEncoderDecoder *allocEncoderDecoder(
    uint enc_vocab_size,
    uint dec_vocab_size,
    uint bos_id,
    uint eos_id,
    uint num_hiddens,
    uint num_blks,
    uint ffn_num_hiddens,
    uint num_heads,
    DATATYPE dropout
) {
    Encoder *encoder = new Encoder(
        enc_vocab_size, num_hiddens, 
        ffn_num_hiddens, num_heads,
        num_blks, dropout
    );
    Decoder *decoder = new Decoder(
        dec_vocab_size, num_hiddens, 
        ffn_num_hiddens, num_heads,
        num_blks, dropout
    );
    Seq2SeqEncoderDecoder *encoder_decoder = new Seq2SeqEncoderDecoder(
        encoder, decoder, bos_id, eos_id
    );
    return encoder_decoder;
}

void releaseEncoderDecoder(Seq2SeqEncoderDecoder *encoder_decoder){
    delete encoder_decoder->get_decoder();
    delete encoder_decoder->get_encoder();
    delete encoder_decoder;
}

/*
    只有在warmUp之后LazyLinear才能正常工作，才能get_parameters
    注意optimizer的初始化时机一定要在warmUp之后
    注意checkpoint的加载时机一定要在warmUp之后
*/
void warmUp(Seq2SeqEncoderDecoder *encoder_decoder) {
    std::vector<std::vector<uint>> src_token_ids = {{0}};
    std::vector<std::vector<uint>> tgt_token_ids = {{0}};
    std::vector<uint> valid_lens = {};
    std::vector<autograd::Node *> enc_out_embs;
    std::vector<autograd::Node *> dec_out_embs;
    encoder_decoder->forward(
        src_token_ids, tgt_token_ids, valid_lens,
        enc_out_embs, dec_out_embs
    );
    autograd::freeAllNodes();
    autograd::freeAllEdges();
    freeTmpMatrix();
}

void train(
    const std::string &corpus,
    const std::string &checkpoint,
    uint epochs,
    DATATYPE dropout,
    DATATYPE lr,
    bool tiny) {
    uint num_steps = NUM_STEPS;
    std::string src_vocab_name = tiny ? SRC_VOCAB_TINY_NAME : SRC_VOCAB_NAME;
    std::string tgt_vocab_name = tiny ? TGT_VOCAB_TINY_NAME : TGT_VOCAB_NAME;
    seq2seq::DataLoader loader(corpus, src_vocab_name, tgt_vocab_name, TEST_FILE);
    std::cout << "data loaded" << std::endl;
    uint enc_vocab_size = loader.src_vocab_size();
    uint dec_vocab_size = loader.tgt_vocab_size();
    uint num_hiddens = tiny ? TINY_NUM_HIDDENS : NUM_HIDDENS;
    uint num_blks = NUM_BLKS;
    uint ffn_num_hiddens = tiny ? TINY_FFN_NUM_HIDDENS: FFN_NUM_HIDDENS;
    uint num_heads = NUM_HEADS;
    Seq2SeqEncoderDecoder *encoder_decoder = allocEncoderDecoder(
        enc_vocab_size, dec_vocab_size,
        loader.tgt_bos_id(), loader.tgt_eos_id(),
        num_hiddens, num_blks, ffn_num_hiddens, num_heads, dropout
    );
    warmUp(encoder_decoder);
    std::cout << "warmUp done" << std::endl;
    auto parameters = encoder_decoder->get_parameters();
    /*
    parameter size = 

    enc_vocab_size + dec_vocab_size // vocab embeddings

    + num_blks * (
        // encoder block
        4 // multi-head attention Wq Wk Wv Wo no bias
        4 // ffn dense1 dense2 with bias
        2 + 2 // addnorm1 gamma beta addnorm2 gamma beta
        // decoder block
        
        4 // self attention
        4 // enc attention
        4 // ffn
        2 + 2 + 2 // addnorm1 addnorm2 addnorm3
    ) + 2 // decoder final linear with bias
    */
    if (
        parameters.size() == 
        enc_vocab_size + dec_vocab_size +
        num_blks * (
            4 + 4 + 2 + 2 + 4 + 4 + 4 + 2 + 2 + 2
        ) + 2
    ) {
        std::cout << "parameter size = " << parameters.size() << std::endl;
    } else {
        std::cerr << "parameter size = " << parameters.size() << std::endl;
        abort();
    }
    for (auto p : parameters) {
        if (!p->require_grad()) {
            std::cerr << "parameter require_grad = false" << std::endl;
            abort();
        }
    }
    std::cout << "all parameters require_grad = true" << std::endl;
    if (!checkpoint.empty()) {
        cout << "loading from checkpoint : " << checkpoint << endl;
        loadfrom_checkpoint(*encoder_decoder, checkpoint);
        cout << "loaded from checkpoint" << endl;
    }
    auto adam = autograd::Adam(parameters, lr);
    std::string checkpoint_prefix = "checkpoint" + generateDateTimeSuffix();
    std::vector<std::vector<uint>> src_token_ids;
    std::vector<std::vector<uint>> tgt_token_ids;
    loader.get_token_ids(src_token_ids, tgt_token_ids);
    assert(src_token_ids.size() == tgt_token_ids.size());
    for (uint epoch = 0; epoch < epochs; epoch++) {
        DATATYPE loss_sum = 0;
        int emit_clip = 0;
        int cnt = 0;
        print_progress(0, src_token_ids.size());
        for (uint i = 0; i < src_token_ids.size(); i += BATCH_SIZE) {
            cnt ++;
            std::vector<std::vector<uint>> input_sentences;
            std::vector<std::vector<uint>> target_sentences;
            std::vector<std::vector<uint>> target_labels;
            std::vector<uint> labels;
            std::vector<bool> mask;
            std::vector<uint> enc_valid_lens;
            auto end = i + BATCH_SIZE;
            if (end > src_token_ids.size()) {
                end = src_token_ids.size();
            }
            auto cur_batch_size = end - i;
            for (uint j = i; j < end; j++) {
                enc_valid_lens.push_back(src_token_ids[j].size());
                input_sentences.push_back(trim_or_padding(src_token_ids[j], num_steps, loader.src_pad_id()));
                target_sentences.push_back(trim_or_padding(add_bos(tgt_token_ids[j], loader.tgt_bos_id()), num_steps, loader.tgt_pad_id()));
                target_labels.push_back(trim_or_padding(tgt_token_ids[j], num_steps, loader.tgt_pad_id()));
            }
            for (auto &l : target_labels) {
                for (auto &t : l) {
                    labels.push_back(t);
                    mask.push_back(t != loader.tgt_pad_id());
                }
            }
            assert(input_sentences.size() == cur_batch_size);
            assert(target_sentences.size() == cur_batch_size);
            assert(target_labels.size() == cur_batch_size);
            assert(labels.size() == cur_batch_size * num_steps);
            assert(mask.size() == cur_batch_size * num_steps);
            std::vector<autograd::Node *> enc_out_embs;
            std::vector<autograd::Node *> dec_out_embs;
            auto dec_outputs = encoder_decoder->forward(
                input_sentences, target_sentences, enc_valid_lens,
                enc_out_embs, dec_out_embs
            );
            dec_outputs->checkShape(Shape(dec_vocab_size, cur_batch_size * num_steps));
            auto loss = dec_outputs->CrossEntropyMask(labels, mask);
            assert(loss->get_weight()->getShape().rowCnt == 1);
            assert(loss->get_weight()->getShape().colCnt == 1);
            loss_sum += (*loss->get_weight())[0][0];
            adam.zero_grad();
            loss->backward();
            if (adam.clip_grad(1)) {
                emit_clip++;
            }
            adam.step();
            print_progress(end, src_token_ids.size());
            if (shutdown) {
                save_checkpoint(checkpoint_prefix, epoch, *encoder_decoder);
                exit(0);
            }
            freeTmpMatrix();
            autograd::freeAllNodes();
            autograd::freeAllEdges();
        }
        if (epoch % 10 == 0 || epoch == epochs - 1) {
            save_checkpoint(checkpoint_prefix, epoch, *encoder_decoder);
        }
        std::cout << "epoch " << epoch << " loss : " << loss_sum/cnt << " emit_clip : " << emit_clip << std::endl;
    }

    if (epochs == 0) {
        std::cout << "serving mode" << std::endl;
        autograd::dropout_run = false;
        encoder_decoder->train(false);
        std::vector<std::string> src_sentences = loader.get_test_sentences();
        for (auto & sentence : src_sentences) {
            std::vector<uint> src_token_ids = loader.to_src_token_ids(sentence);
            for (auto &token_id : src_token_ids) {
                std::cout << loader.get_src_token(token_id) << " ";
            }
            std::cout << std::endl;
            std::vector<autograd::Node *> enc_out_embs;
            std::vector<autograd::Node *> dec_out_embs;
            std::vector<uint> tgt_token_ids = encoder_decoder->predict(
                src_token_ids, 20,
                enc_out_embs, dec_out_embs
            );
            if (autograd::dropout_run) {
                std::cerr << "[warning!!!!] dropout run" << std::endl;
            } 
            std::cout << "translate res : ";
            for (auto &token_id : tgt_token_ids) {
                std::cout << loader.get_tgt_token(token_id) << " ";
            }
            std::cout << std::endl;
        }
    }
    freeTmpMatrix();
    autograd::freeAllNodes();
    autograd::freeAllEdges();
    releaseEncoderDecoder(encoder_decoder);
}

int main(int argc, char *argv[]) {
    cout << "OMP_THREADS: " << OMP_THREADS << endl;
    // test();
    // return -1;
    signal(SIGINT, signal_callback_handler);
    std::string corpus;
    std::string checkpoint;
    int opt;
    uint epochs = 30;
    DATATYPE dropout = 0.2;
    DATATYPE lr = 0.001;
    bool tiny = false;
    while ((opt = getopt(argc, argv, "f:c:e:d:l:t:")) != -1) {
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
            case 'd':
                dropout = atof(optarg);
                break;
            case 'l':
                lr = atof(optarg);
                break;
            case 't':
                tiny = atoi(optarg) == 1;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -f <corpus> -c <checpoint> -e <epochs>" << std::endl;
                return 1;
        }
    }
    if (corpus.empty()) {
        corpus = RESOURCE_NAME;
    }
    std::cout << "epochs : " << epochs << std::endl;
    std::cout << "dropout : " << dropout << std::endl;
    std::cout << "lr : " << lr << std::endl;
    std::cout << "tiny : " << tiny << std::endl;
    train(corpus, checkpoint, epochs, dropout, lr, tiny);
    return 0;
}