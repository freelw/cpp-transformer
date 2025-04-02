def get_fra():
    with open("./fra.txt") as f:
        return f.read()

def preprocess(text):
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Insert space between words and punctuation marks
    no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text.lower())]
    return ''.join(out)

if '__main__' == __name__:

    with open("./fra_preprocessed.txt", "w") as f:
        f.write(preprocess(get_fra()))