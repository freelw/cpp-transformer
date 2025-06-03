#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "optimizers/parameter.h"


std::string generateDateTimeSuffix();
void save_checkpoint(const std::string& prefix, int epoch, const std::vector<Parameter*>& parameters);
void loadfrom_checkpoint(const std::string& filename, std::vector<Parameter*>& parameters);

#endif