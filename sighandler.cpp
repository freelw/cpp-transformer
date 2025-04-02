// create a signal handler for the program


#include <iostream>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>

extern bool shutdown;

void signal_callback_handler(int signum) {
    // std::cout << "Caught signal " << signum << std::endl;
    shutdown = true;
}