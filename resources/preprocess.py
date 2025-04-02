import re

def get_timemachine():
    with open("./timemachine.txt") as f:
        return f.read()

def preprocess(text):
    """Defined in :numref:`sec_text-sequence`"""
    return re.sub('[^A-Za-z]+', ' ', text).lower()

if '__main__' == __name__:

    with open("./timemachine_preprocessed.txt", "w") as f:
        f.write(preprocess(get_timemachine()))