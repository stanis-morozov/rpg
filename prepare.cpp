#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <algorithm>

const int DEFAULT_TRAINSIZE = 1000;
const int DEFAULT_TESTSIZE = 1000;
const int DEFAULT_TOPSIZE = 100;

void printHelpMessage()
{
    std::cout << "Usage: prepare [OPTIONS]" << std::endl;
    std::cout << "This tool converts a database of euclidean vectors into RPG-compatible format." << std::endl;
    std::cout << std::endl;


    std::cout << "  --basesize     " << "Use first basesize items from the base file" << std::endl;
    std::cout << "  --base         " << "Base filename. Should be a set of vector in *.fvecs format" << std::endl;
    std::cout << "  --querysize    " << "Use first querysize queries" << std::endl;
    std::cout << "  --query        " << "Base filename. Should be a set of vector in *.fvecs format" << std::endl;
    std::cout << "  --trainsize    " << "Number of train vectors (maximal value, you can use less)" << std::endl;
    std::cout << "  --testsize     " << "Number of test vector (maximal value, you can use less)" << std::endl;
    std::cout << "  --topsize      " << "Top size to compute groundtruth (maximal value, you can use less)" << std::endl;
    std::cout << "  --outfolder    " << "Folder to save output" << std::endl;
    std::cout << "  --suffix       " << "Suffix to add to the produces filenames" << std::endl;
}


int main(int argc, char *argv[])
{
    std::vector <float> base;
    std::vector <float> query;

    std::vector <float> distances;
    std::vector <int> gt;

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--h" || std::string(argv[i]) == "--help") {
           printHelpMessage();
           return 0; 
        }
    }
    
    int basesize = -1;
    int dim = -1;
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--basesize") {
            if (sscanf(argv[i + 1], "%d", &basesize) != 1 || basesize <= 0) {
                std::cerr << "Inappropriate value for base size: " << argv[i + 1] << std::endl;
                return 1;
            }
            break;
        }
    }
    if (basesize == -1) {
        std::cerr << "Base size was not specified" << std::endl;
        return 1;
    }
   
    bool good_base = false; 
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--base") {
            std::ifstream base_file(argv[i + 1], std::ios::binary);
            if (!base_file.is_open()) {
                std::cerr << "No such file " << argv[i + 1] << std::endl;
                return 1;
            }
            good_base = true;
            for (int i = 0; i < basesize; i++) {
                int sz;
                base_file.read((char*)&sz, sizeof(sz));
                if (i == 0) {
                    base.resize((long long)basesize * sz);
                    if (dim == -1) {
                        dim = sz;
                    }
                }
                assert(dim == sz);
                base_file.read((char*)(base.data() + i * sz), sizeof(float) * sz);
            }
            base_file.close();

            break;
        }
    }
    if (!good_base) {
        std::cerr << "Base file was not specified" << std::endl;
        return 1;
    }
    

    int querysize = -1;
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--querysize") {
            if (sscanf(argv[i + 1], "%d", &querysize) != 1 || querysize <= 0) {
                std::cerr << "Inappropriate value for number of queries: " << argv[i + 1] << std::endl;
                return 1;
            }
            break;
        }
    }
    if (querysize == -1) {
        std::cerr << "Number of queries was not specified" << std::endl;
        return 1;
    }
   
    bool good_query = false; 
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--query") {
            std::ifstream query_file(argv[i + 1], std::ios::binary);
            if (!query_file.is_open()) {
                std::cerr << "No such file " << argv[i + 1] << std::endl;
                return 1;
            }
            good_query = true;
            for (int i = 0; i < querysize; i++) {
                int sz;
                query_file.read((char*)&sz, sizeof(sz));
                if (i == 0) {
                    query.resize((long long)querysize * sz);
                    if (dim == -1) {
                        dim = sz;
                    }
                }
                assert(dim == sz);
                query_file.read((char*)(query.data() + i * sz), sizeof(float) * sz);
            }
            query_file.close();

            break;
        }
    }
    if (!good_query) {
        std::cerr << "Query file was not specified" << std::endl;
        return 1;
    }

    int trainsize = DEFAULT_TRAINSIZE;
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--trainsize") {
            if (sscanf(argv[i + 1], "%d", &trainsize) != 1 || trainsize <= 0) {
                std::cerr << "Inappropriate value for train size: " << argv[i + 1] << std::endl;
                return 1;
            }
            break;
        }
    }
    
    int testsize = DEFAULT_TESTSIZE;
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--testsize") {
            if (sscanf(argv[i + 1], "%d", &testsize) != 1 || testsize <= 0) {
                std::cerr << "Inappropriate value for test size: " << argv[i + 1] << std::endl;
                return 1;
            }
            break;
        }
    }
    
    int topsize = DEFAULT_TOPSIZE;
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--topsize") {
            if (sscanf(argv[i + 1], "%d", &topsize) != 1 || topsize <= 0) {
                std::cerr << "Inappropriate value for top size: " << argv[i + 1] << std::endl;
                return 1;
            }
            break;
        }
    }
    
    std::string suffix;
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--suffix") {
            suffix = argv[i + 1];
            break;
        }
    }
    
    std::string outfolder;
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--outfolder") {
            outfolder = argv[i + 1];
            break;
        }
    }
    if (!outfolder.empty()) {
        outfolder += "/";
    }


    distances.resize((long long)basesize * querysize);
    gt.resize((long long)testsize * topsize);   
    
    #pragma omp parallel for
    for (int i = 0; i < testsize; i++) {
        std::vector <std::pair <float, int> > cand;
        for (int j = 0; j < basesize; j++) {
            float val = 0;
            for (int t = 0; t < dim; t++) {
                float tmp = base[(long long)j * dim + t] - query[(long long)(i + trainsize) * dim + t];
                val += tmp * tmp;
            }
            cand.push_back(std::make_pair(val, j));
        }
        std::sort(cand.begin(), cand.end());
        for (int t = 0; t < topsize; t++) {
            gt[i * topsize + t] = cand[t].second;
        }
    }
 
    #pragma omp parallel for
    for (int i = 0; i < querysize; i++) {
        for (int j = 0; j < basesize; j++) {
            float dst = 0;
            for (int k = 0; k < dim; k++) {
                float tmp = query[i * dim + k] - base[j * dim + k];
                dst += tmp * tmp;
            }
            distances[(long long)j * querysize + i] = -dst;
        }
    }
    
    std::ofstream train_file;
    std::ofstream test_file;
    std::ofstream gt_file;
    if (suffix.empty()) {
        train_file.open(outfolder + "train.bin", std::ios::binary);
        test_file.open(outfolder + "test.bin", std::ios::binary);
        gt_file.open(outfolder + "groundtruth.bin", std::ios::binary);
    } else {
        train_file.open(outfolder + "train_" + suffix + ".bin", std::ios::binary);
        test_file.open(outfolder + "test_" + suffix + ".bin", std::ios::binary);
        gt_file.open(outfolder + "groundtruth_" + suffix + ".bin", std::ios::binary);
    }

    gt_file.write((char*)gt.data(), sizeof(int) * (long long)testsize * topsize);
    for (int i = 0; i < basesize; i++) {
        train_file.write((char*)(distances.data() + (long long)i * querysize), sizeof(distances[0]) * trainsize);
        test_file.write((char*)(distances.data() + (long long)i * querysize + trainsize), sizeof(distances[0]) * testsize);
    }
    train_file.close();
    test_file.close();
    gt_file.close();

    return 0;
}
