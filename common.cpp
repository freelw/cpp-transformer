#include "common.h"
#include "optimizers/parameter.h"

BackendOps *g_backend_ops = nullptr;
bool g_training = true;

bool b_use_gpu = false;

void zero_grad() {
    gCreateAction(
        new ZeroGradAction()
    );
}

void zero_c_tensors() {
    gCreateAction(
        new ZeroCTensorsAction()
    );
}

void print_no_zero_tensor_names() {
    gCreateAction(
        new PrintNoZeroTensorNamesAction()
    );
}

void insert_boundary_action() {
    gCreateAction(
        new BoundaryAction()
    );
}

void init_backend() {
    if (b_use_gpu) {
        #ifndef GCC_CPU
        g_backend_ops = new CUDAOps();
        #else
        std::cerr << "Warning: GPU backend is not available in CPU build. Now use cpu instead!!!" << std::endl;
        g_backend_ops = new CPUOps();
        #endif
    } else {
        g_backend_ops = new CPUOps();
    }
}

void release_backend() {
    delete g_backend_ops;
    g_backend_ops = nullptr;
}

void construct_env() {
    init_backend();
}

void destruct_env() {
    // sanitizeTensors();
    releaseParameters();
    freeAllActions();
    freeAllTensors();
    freeAllCTensors();
    freeAllTensorViews();
    freeAllGradTensors();
    graph::freeAllNodes();
    graph::freeAllEdges();
    releaseTensorMem();
    release_backend();
}

void use_gpu(bool use) {
    b_use_gpu = use;
}

bool is_use_gpu() {
    return b_use_gpu;
}

void print_all_tensors() {
    //print all tensors
    for (int i = 0; i < g_c_tensors.size(); i++) {
        std::cout << "c_tensor " << i << " name : " << g_c_tensors[i]->get_meta_info() << std::endl;
        std::cout << "c_tensor " << i << " value : " << std::endl << *g_c_tensors[i] << std::endl;
    }
    for (int i = 0; i < g_grad_tensors.size(); i++) {
        std::cout << "grad_tensor " << i << " name : " << g_grad_tensors[i]->get_meta_info() << std::endl;
        std::cout << "grad_tensor " << i << " value : " << std::endl << *g_grad_tensors[i] << std::endl;
    }
    for (int i = 0; i < g_tensors.size(); ++ i) {
        std::cout << "tensor " << i << " name : " << g_tensors[i]->get_meta_info() << std::endl;
        std::cout << "tensor " << i << " value : " << std::endl << *g_tensors[i] << std::endl;
    }
    for (int i = 0; i < g_tensor_views.size(); ++ i) {
        std::cout << "tensor_view " << i << " name : " << g_tensor_views[i]->get_meta_info() << std::endl;
        std::cout << "tensor_view " << i << " value : " << std::endl << *g_tensor_views[i] << std::endl;
    }
}