#include "common.h"
#include "checkpoint.h"
#include "dataloaders/language_model/lm_dataloader.h"
#include "module/language_model/lm_decoder.h"
#include "optimizers/adam.h"
#include <unistd.h>
#include <signal.h>
#include <sstream>

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
    LMDataLoader& loader,
    std::vector<std::vector<uint>>& v_src_token_ids,
    std::vector<std::vector<uint>>& v_tgt_token_ids,
    int& dec_vocab_size,
    int& pad_id
) {
    loader.get_token_ids(v_src_token_ids, v_tgt_token_ids);
    dec_vocab_size = loader.tgt_vocab_size();
    pad_id = loader.get_pad_id();
}

void init_dec_valid_lens_for_training(Tensor* dec_valid_lens) {
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

void init_dec_valid_lens_for_predict(Tensor* dec_valid_lens, int cur_step) {
    auto shape = dec_valid_lens->get_shape();
    assert(shape.size() == 1 && shape[0] == 1);
    g_backend_ops->cp_to_device(
        dec_valid_lens,
        reinterpret_cast<char*>(&cur_step),
        dec_valid_lens->size()
    );
}

int main(int argc, char* argv[]) {

    shutdown = false;

    int opt;
    int epochs = 10;
    int batch_size = 16;
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
    int num_steps = LM_NUM_STEPS;
    int max_posencoding_len = MAX_POSENCODING_LEN;

    std::string tgt_vocab_name = TIMEMACHINE_VOCAB_NAME;
    std::string test_file = TEST_LM_FILE;
    LMDataLoader loader(corpus, tgt_vocab_name, test_file, num_steps);

    int dec_vocab_size = 0;
    int pad_id = 0;

    std::vector<std::vector<uint>> v_tgt_token_ids;
    std::vector<std::vector<uint>> v_src_token_ids;
    load_tokens_from_file(
        loader,
        v_src_token_ids,
        v_tgt_token_ids,
        dec_vocab_size,
        pad_id
    );

    // for (int i = 0; i < 100; ++i) {
    //     for (int j = 0; j < v_src_token_ids[i].size(); ++j) {
    //         std::cout << loader.get_tgt_token(v_src_token_ids[i][j]) << " ";
    //     }
    //     std::cout << " -> ";
    //     for (int j = 0; j < v_tgt_token_ids[i].size(); ++j) {
    //         std::cout << loader.get_tgt_token(v_tgt_token_ids[i][j]) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // exit(0);

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

    Tensor* tgt_token_ids = predicting ? allocTensor({ batch_size, num_steps }, INT32) : allocTensor({ batch_size * num_steps, num_steps }, INT32);
    Tensor* dec_valid_lens = predicting ? allocTensor({ 1 }, INT32) : allocTensor({ batch_size * num_steps, num_steps }, INT32);
    Tensor* labels = allocTensor({ batch_size * num_steps * num_steps }, INT32);
    Tensor* ce_mask = allocTensor({ batch_size * num_steps * num_steps });

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
        assert(batch_size == 1);
        std::cout << "serving mode" << std::endl;
        std::cout << "test file : " << test_file << std::endl;
        std::vector<std::string> src_sentences = loader.get_test_sentences();
        for (auto& sentence : src_sentences) {
            std::cout << "sentence : " << sentence << std::endl;
            std::vector<uint> src_token_ids;
            std::istringstream iss(sentence);
            std::string token;
            while (iss >> token) {
                // std::cout << "token : " << token << std::endl;
                src_token_ids.push_back(loader.get_tgt_token_id(token));
            }
            auto origin_size = src_token_ids.size();
            // for (int i = 0; i < origin_size; ++i) {
            //     std::cout << loader.get_tgt_token(src_token_ids[i]) << " ";
            // }
            // std::cout << std::endl;
            if (src_token_ids.size() < num_steps) {
                src_token_ids.resize(num_steps, loader.get_pad_id());
            } else if (src_token_ids.size() > num_steps) {
                src_token_ids.erase(src_token_ids.begin(), src_token_ids.end() - num_steps);
            }
            auto cur_step = origin_size - 1;
            float* res_buffer = static_cast<float*>(::malloc(
                res->get_tensor()->size()
            ));
            for (int i = 0; i < LM_PREDICT_CNT; ++i) {

                for (int j = 0; j < num_steps; ++j) {
                    tgt_token_ids_buffer[j] = src_token_ids[j];
                    std::cout << loader.get_tgt_token(src_token_ids[j]) << " ";
                }
                std::cout << std::endl;
                init_dec_valid_lens_for_predict(dec_valid_lens, cur_step + 1);

                if (cur_step < num_steps - 1) {
                    cur_step++;
                }
                std::cout << "cur_step: " << cur_step << std::endl;

                g_backend_ops->cp_to_device(
                    tgt_token_ids,
                    reinterpret_cast<char*>(tgt_token_ids_buffer),
                    tgt_token_ids->size()
                );
                gDoForwardActions();

                g_backend_ops->cp_from_device(
                    reinterpret_cast<char*>(res_buffer),
                    res->get_tensor(),
                    res->get_tensor()->size()
                );
                assert(res->get_tensor()->length() == dec_vocab_size * num_steps);
                int offset = (cur_step - 1) * dec_vocab_size;
                int max_index = 0;
                float max_value = res_buffer[offset];
                for (int i = 0; i < loader.tgt_vocab_size(); ++i) {
                    if (res_buffer[offset + i] > max_value) {
                        max_value = res_buffer[offset + i];
                        max_index = i;
                    }
                }

                std::cout << loader.get_tgt_token(max_index) << " ";
                if (cur_step >= num_steps - 1) {
                    src_token_ids.push_back(max_index);
                    src_token_ids.erase(src_token_ids.begin(), src_token_ids.end() - num_steps);
                } else {
                    src_token_ids[cur_step] = max_index;
                }
            }
            std::cout << std::endl;
            std::cout << "-----------------" << std::endl;
            ::free(res_buffer);
            // src_token_ids.erase(src_token_ids.begin(), src_token_ids.end() - num_steps);
            // for (auto& token_id : src_token_ids) {
            //     std::cout << loader.get_tgt_token(token_id) << " ";
            // }
            // std::cout << std::endl;
        }
    } else {
        init_dec_valid_lens_for_training(dec_valid_lens);
        signal(SIGINT, signal_callback_handler);
        int epoch = 0;
        for (; epoch < epochs; ++epoch) {
            if (shutdown) {
                break;
            }
            float loss_sum = 0;
            int cnt = 0;
            std::string prefix = "epoch " + std::to_string(epoch) + " : ";
            for (int i = 0; i + num_steps < v_tgt_token_ids.size(); i += batch_size) {
                if (shutdown) {
                    break;
                }
                cnt++;
                auto end = i + batch_size;
                if (end > v_src_token_ids.size()) {
                    break;
                }
                for (int j = i; j < end; ++j) {
                    for (int len = 0; len < num_steps; ++len) {
                        for (int k = 0; k < num_steps; ++k) {
                            auto base = (j - i) * num_steps * num_steps + len * num_steps;
                            tgt_token_ids_buffer[base + k] = v_src_token_ids[j - i][k];
                            labels_buffer[base + k] = v_tgt_token_ids[j - i][k];
                            ce_mask_buffer[base + k] = (k <= len) ? 1.0f : 0.0f;
                        }
                    }
                }
                // for (int j = i; j < end; ++j) {
                //     for (int len = 0; len < num_steps; ++len) {
                //         auto base = (j - i) * num_steps * num_steps + len * num_steps;
                //         for (int k = 0; k < num_steps; ++k) {
                //             std::cout << tgt_token_ids_buffer[base + k] << " ";
                //         }
                //         std::cout << std::endl;
                //         for (int k = 0; k < num_steps; ++k) {
                //             std::cout << labels_buffer[base + k] << " ";
                //         }
                //         std::cout << std::endl;
                //         for (int k = 0; k < num_steps; ++k) {
                //             std::cout << ce_mask_buffer[base + k] << " ";
                //         }
                //         std::cout << std::endl;
                //     }
                // }
                // exit(0);


                g_backend_ops->cp_to_device(
                    tgt_token_ids,
                    reinterpret_cast<char*>(tgt_token_ids_buffer),
                    tgt_token_ids->size()
                );
                g_backend_ops->cp_to_device(
                    labels,
                    reinterpret_cast<char*>(labels_buffer),
                    labels->size()
                );
                g_backend_ops->cp_to_device(
                    ce_mask,
                    reinterpret_cast<char*>(ce_mask_buffer),
                    ce_mask->size()
                );
                gDoActions();
                print_progress(prefix, end, v_src_token_ids.size());
                float loss_v = 0;
                g_backend_ops->cp_from_device(
                    reinterpret_cast<char*>(&loss_v),
                    loss->get_tensor(),
                    loss->get_tensor()->size()
                );
                loss_sum += loss_v;
            }
            std::cout << "loss : " << loss_sum / cnt << std::endl;
        }
        std::string checkpoint_prefix = "checkpoint" + generateDateTimeSuffix();
        save_checkpoint(checkpoint_prefix, shutdown ? epoch : epoch - 1, parameters);
    }
    ::free(tgt_token_ids_buffer);
    ::free(labels_buffer);
    ::free(ce_mask_buffer);
    delete lm_decoder;
    destruct_env();
    return 0;
}