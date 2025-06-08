#include "common.h"
#include "checkpoint.h"
#include "dataloader.h"
#include "module/translation/seq2seq.h"
#include "optimizers/adam.h"
#include <unistd.h>
#include <signal.h>

extern bool shutdown;
void signal_callback_handler(int signum);

int main(int argc, char* argv[]) {

    shutdown = false;

    int opt;
    int epochs = 10;
    int batch_size = 128;
    int gpu = 1;
    float lr = 0.001f;
    std::string checkpoint;
    std::string corpus = RESOURCE_NAME;

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
    return 0;
}