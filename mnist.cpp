#include <iostream>
#include <unistd.h>
#include "common.h"
#include "tensor.h"
#include "backends/backend_ops.h"
#include "graph/node.h"
#include "dataloaders/mnist/mnist_loader_base.h"
#include "optimizers/adam.h"
#include "model/mlp.h"
#include <string.h>

#define INPUT_LAYER_SIZE 784

void print_progress(const std::string& prefix, uint i, uint tot) {
    std::cout << "\r" << prefix << " [" << i << "/" << tot << "]" << std::flush;
}

void assign_inputs(
    Tensor* inputs, float* tmp_buffer,
    int offset,
    int batch_size,
    const std::vector<std::vector<unsigned char>>& data) {
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < INPUT_LAYER_SIZE; ++j) {
            tmp_buffer[i * INPUT_LAYER_SIZE + j] = static_cast<float>(data[offset + i][j]);
        }
    }
    g_backend_ops->cp_to_device(
        inputs,
        reinterpret_cast<char*>(tmp_buffer),
        inputs->size()
    );
}

void assign_labels(
    Tensor* labels, int32_t* tmp_buffer,
    int offset,
    int batch_size,
    const std::vector<unsigned char>& train_labels) {

    assert(labels->get_dtype() == INT32);
    for (int i = 0; i < batch_size; ++i) {
        tmp_buffer[i] = static_cast<int32_t>(train_labels[offset + i]);
    }
    g_backend_ops->cp_to_device(
        labels,
        reinterpret_cast<char*>(tmp_buffer),
        labels->size()
    );
}

void train(int epochs, float lr, int batch_size) {
    MnistLoaderBase loader;
    loader.load();
    std::cout << "data loaded." << std::endl;

    zero_c_tensors();
    zero_grad();
    MLP m(INPUT_LAYER_SIZE, { 30, 10 }, -0.01f);
    Tensor* inputs = allocTensor({ batch_size, 784 }, "inputs");
    Tensor* labels = allocTensor({ batch_size }, "labels", INT32);
    auto n_inputs = graph::allocNode(inputs);
    auto forward_res = m.forward(n_inputs);
    auto loss = forward_res->CrossEntropy(labels)->avg_1d();
    assert(loss->get_tensor()->size() == sizeof(float));
    insert_boundary_action();
    loss->backward();
    Adam optimizer(m.get_parameters(), lr);
    optimizer.clip_grad(1.0f);
    optimizer.step();
    printAllActions();
    printDotGraph();
    allocMemAndInitTensors();
    float* inputs_tmp_buffer = static_cast<float*>(::malloc(inputs->size()));
    int32_t* labels_tmp_buffer = static_cast<int32_t*>(::malloc(labels->size()));
    float* evaluate_tmp_buffer = static_cast<float*>(::malloc(forward_res->get_tensor()->size()));
    const std::vector<std::vector<unsigned char>>& train_images = loader.getTrainImages();
    const std::vector<unsigned char>& train_labels = loader.getTrainLabels();
    assert(train_images.size() % batch_size == 0);
    assert(TRAIN_IMAGES_NUM + TEST_IMAGES_NUM == train_images.size());
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float loss_sum = 0;
        int offset = 0;
        int loop_times = 0;
        std::string prefix = "epoch : " + std::to_string(epoch);
        print_progress(prefix, offset, TRAIN_IMAGES_NUM);

        while (offset < TRAIN_IMAGES_NUM) {
            assign_inputs(
                inputs,
                inputs_tmp_buffer,
                offset,
                batch_size,
                train_images
            );
            assign_labels(
                labels,
                labels_tmp_buffer,
                offset,
                batch_size,
                train_labels
            );
            offset += batch_size;
            gDoActions();
            float loss_val = 0;
            g_backend_ops->cp_from_device(
                reinterpret_cast<char*>(&loss_val),
                loss->get_tensor(),
                loss->get_tensor()->size()
            );
            loss_sum += loss_val;
            loop_times++;
            print_progress(prefix, offset, TRAIN_IMAGES_NUM);
        }
        std::cout << " loss : " << loss_sum / loop_times << std::endl;

        // evaluate
        offset = TRAIN_IMAGES_NUM;
        print_progress("evaluating :", offset - TRAIN_IMAGES_NUM, TEST_IMAGES_NUM);
        int correct = 0;
        while (offset < TRAIN_IMAGES_NUM + TEST_IMAGES_NUM) {
            assign_inputs(
                inputs,
                static_cast<float*>(inputs_tmp_buffer),
                offset,
                batch_size,
                train_images
            );

            gDoForwardActions();
            g_backend_ops->cp_from_device(
                reinterpret_cast<char*>(evaluate_tmp_buffer),
                forward_res->get_tensor(),
                forward_res->get_tensor()->size()
            );
            for (int i = 0; i < batch_size; ++i) {
                int max_index = 0;
                float max_value = evaluate_tmp_buffer[i * 10];
                for (int j = 1; j < 10; ++j) {
                    if (evaluate_tmp_buffer[i * 10 + j] > max_value) {
                        max_value = evaluate_tmp_buffer[i * 10 + j];
                        max_index = j;
                    }
                }
                if (max_index == static_cast<int>(train_labels[offset + i])) {
                    correct++;
                }
            }
            offset += batch_size;
            print_progress("evaluating : ", offset - TRAIN_IMAGES_NUM, TEST_IMAGES_NUM);
        }
        std::cout << " correct : " << correct << std::endl;
    }
    ::free(evaluate_tmp_buffer);
    ::free(labels_tmp_buffer);
    ::free(inputs_tmp_buffer);
}

int main(int argc, char* argv[]) {
#ifdef GCC_ASAN
#pragma message "GCC_ASAN"
#endif
    int opt;
    int epochs = 10;
    int batch_size = 100;
    int gpu = 1;
    float lr = 0.001f;

    while ((opt = getopt(argc, argv, "e:l:b:g:")) != -1) {
        switch (opt) {
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
            std::cerr << "Usage: " << argv[0] << " -f <corpus> -c <checpoint> -e <epochs>" << std::endl;
            return 1;
        }
    }
    use_gpu(gpu == 1);
    construct_env();
    if (epochs > 0) {
        train(epochs, lr, batch_size);
    }
    else {
        // serving
    }
    destruct_env();
    return 0;
}