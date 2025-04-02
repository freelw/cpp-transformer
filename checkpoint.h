#ifndef TRANSFORMER_CHECKPOINT_H
#define TRANSFORMER_CHECKPOINT_H

#include <string>
#include "seq2seq.h"

std::string generateDateTimeSuffix();
void save_checkpoint(const std::string & prefix, int epoch, Seq2SeqEncoderDecoder &lm);
void loadfrom_checkpoint(Seq2SeqEncoderDecoder &lm, const std::string &filename);

#endif