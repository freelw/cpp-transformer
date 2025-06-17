#include <iostream>
#include <vector>
#include <map>
#include <fstream>

using namespace std;

#define RESOURCE_NAME "../../resources/time_machine/timemachine_preprocessed.txt"

int main() {
    ifstream file(RESOURCE_NAME);
    std::vector<std::string> words;
    std::string word;
    while (file >> word) {
        words.push_back(word);
    }

    std::map<std::string, uint> vocab;
    for (auto w : words) {
        if (vocab.find(w) == vocab.end()) {
            vocab[w] = 1;
        }
        else {
            vocab[w]++;
        }
    }

    for (auto it = vocab.begin(); it != vocab.end(); it++) {
        cout << it->first << " : " << it->second << endl;
    }

    // show distribution
    std::map<uint, uint> dist;
    int unknown_cnt = 0;
    int unknown_cnt2 = 0;
    for (auto it = vocab.begin(); it != vocab.end(); it++) {
        if (it->second < 1) {
            unknown_cnt++;
            unknown_cnt2 += it->second;
            continue;
        }
        if (dist.find(it->second) == dist.end()) {
            dist[it->second] = 1;
        }
        else {
            dist[it->second]++;
        }
    }
    for (auto it = dist.begin(); it != dist.end(); it++) {
        cout << it->first << " : " << it->second << endl;
    }
    std::cout << "unknown_cnt : " << unknown_cnt << std::endl;
    std::cout << "unknown_cnt2 : " << unknown_cnt2 << std::endl;
    std::cout << "dist size : " << dist.size() << std::endl;
    std::cout << "words size : " << words.size() << std::endl;
    std::cout << "vocab size : " << vocab.size() << std::endl;

    // save vocab
    std::ofstream out("vocab.txt");
    for (auto it = vocab.begin(); it != vocab.end(); it++) {
        out << it->first << std::endl;
    }
    out.close();

    return 0;
}