def get_fra_preprocessed():
    with open("../../resources/fra_preprocessed.txt") as f:
        return f.read()

if '__main__' == __name__:

    content = get_fra_preprocessed()
    # for eache line in content, split the line into two parts
    # The first part is the English sentence, and the second part is the French sentence
    # The two sentences are separated by a tab character

    vocab_en = dict()
    vocab_fr = dict()
    
    for line in content.split('\n'):
        # Split the line into two parts
        arr = line.split('\t')
        if len(arr) != 2:
            continue
        en, fr = arr
        # Tokenize the English sentence
        en_tokens = en.split()
        # Tokenize the French sentence
        fr_tokens = fr.split()
        # Print the English sentence
        #print(en_tokens)
        # Print the French sentence
        #print(fr_tokens)
        # Print a blank line
        #print()
        # Break the loop
        # break
        # Update the English vocabulary
        for token in en_tokens:
            if token not in vocab_en:
                vocab_en[token] = 1
            else:
                vocab_en[token] += 1
        # Update the French vocabulary
        for token in fr_tokens:
            if token not in vocab_fr:
                vocab_fr[token] = 1
            else:
                vocab_fr[token] += 1
    # Print the English vocabulary
    # print(vocab_en)

    filterd_vocab_en = {k: v for k, v in vocab_en.items() if v >= 2}
    print(len(vocab_en))
    print(len(filterd_vocab_en))

    # Print the French vocabulary
    #print(vocab_fr)
    filterd_vocab_fr = {k: v for k, v in vocab_fr.items() if v >= 2}
    print(len(vocab_fr))
    print(len(filterd_vocab_fr))

    with open("./vocab_en.txt", "w") as f:
        for token in filterd_vocab_en:
            f.write(token + '\n')
    with open("./vocab_fr.txt", "w") as f:
        for token in filterd_vocab_fr:
            f.write(token + '\n')