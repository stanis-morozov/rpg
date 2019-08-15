#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <queue>
#include <chrono>
#include "hnswlib.h"
#include <algorithm>
#include <ctime>
#include <omp.h>


const int defaultEfConstruction = 1024;
const int defaultEfSearch = -1;
const int defaultM = 8;
const int defaultTopK = 1;

void printHelpMessage()
{
    std::cerr << "Usage: RPG [OPTIONS]" << std::endl;
    std::cerr << "This tool supports two modes: to construct the Relevance Proximity Graph index from the database and to retrieve K most relevant items using the constructed index. Each mode has its own set of parameters." << std::endl;
    std::cerr << std::endl;

    std::cerr << "  --mode             " << "\"base\" or \"query\". Use \"base\" for" << std::endl;
    std::cerr << "                     " << "constructing index and \"query\" for retrieving" << std::endl;
    std::cerr << "                     " << "the most relevant items" << std::endl;
    std::cerr << std::endl;

    std::cerr << "Base mode supports the following options:" << std::endl;
    std::cerr << "  --baseSize         " << "Number of items in the base" << std::endl;
    std::cerr << "  --trainQueries     " << "Number of train queries in the base" << std::endl;
    std::cerr << "  --base             " << "Training base file. Should be a binary file containing a matrix of size (baseSize, trainQueries)" << std::endl;
    std::cerr << "  --outputGraph      " << "Filename for the output index graph" << std::endl;
    std::cerr << "  --relevanceVector  " << "Relevance vector length" << std::endl;
    std::cerr << "  --efConstruction   " << "efConstruction parameter. Default: " << defaultEfConstruction << std::endl;
    std::cerr << "  --M                " << "M parameter. Default: " << defaultM << std::endl;
    std::cerr << std::endl;
    std::cerr << "Query mode supports the following options:" << std::endl;
    std::cerr << "  --baseSize         " << "Number of items in the base" << std::endl;
    std::cerr << "  --querySize        " << "Number of test queries" << std::endl;
    std::cerr << "  --query            " << "File containing query description. Should be a binary file containing a matrix of size (baseSize, querySize)" << std::endl;
    std::cerr << "  --inputGraph       " << "Filename for the input index graph" << std::endl;
    std::cerr << "  --efSearch         " << "efSearch parameter" << std::endl;
    std::cerr << "  --topK             " << "Top size for retrieval. Default: " << defaultTopK << std::endl;
    std::cerr << "  --output           " << "Filename to print the result. Default: " << "stdout" << std::endl;
    std::cerr << "  --gtQueries        " << "Number of queries in the groundtruth file" << std::endl;
    std::cerr << "  --gtTop            " << "Top size in the groundtruth file" << std::endl;
    std::cerr << "  --groundtruth      " << "Groundtruth file. Should be a binary file containing a matrix of size (gtQueries, gtTop)" << std::endl;
}

void printError(std::string err)
{
    std::cerr << err << std::endl;
    std::cerr << std::endl;
    printHelpMessage();
}

int main(int argc, char *argv[])
{
    std::string mode;
    std::ifstream input;
    std::ifstream inputQ;
    int efConstruction = defaultEfConstruction;
    int efSearch = defaultEfSearch;
    int M = defaultM;
    int vecsize = -1;
    int qsize = -1;
    int trainQueries = -1, relevanceVector = -1;
    std::string graphname;
    std::string outputname;
    int topK = defaultTopK;
    std::string base_filename;
    std::string query_filename;
    std::string gt_filename;
    int gtQueries = -1, gtTop = -1;
    bool good_gt;

    hnswlib::HierarchicalNSW<float> *appr_alg;
    
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--h" || std::string(argv[i]) == "--help") {
           printHelpMessage();
           return 0; 
        }
    }
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--mode") {
            if (std::string(argv[i + 1]) == "base") {
                mode = "base";
            } else if (std::string(argv[i + 1]) == "query") {
                mode = "query";
            } else {
                printError("Unknown running mode \"" + std::string(argv[i + 1]) + "\". Please use \"base\" or \"query\"");
                return 0;
            }
            break;
        }
    }
    if (mode.empty()) {
        printError("Running mode was not specified");
        return 0;
    }
    

    std::cout << "Mode: " << mode << std::endl;
        
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--baseSize") {
            if (sscanf(argv[i + 1], "%d", &vecsize) != 1 || vecsize <= 0) {
                printError("Inappropriate value for base size: \"" + std::string(argv[i + 1]) + "\"");
                return 0;
            }
            break;
        }
    }
    if (vecsize == -1) {
        printError("Base size was not specified");
        return 0;
    }
    std::cout << "Base size: " << vecsize << std::endl;

    
    if (mode == "base") {
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--trainQueries") {
                if (sscanf(argv[i + 1], "%d", &trainQueries) != 1 || trainQueries <= 0) {
                    printError("Inappropriate value for the number of train queries: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        if (trainQueries == -1) {
            printError("The number of train queries was not specified");
            return 0;
        }
        std::cout << "Number of train queries: " << trainQueries << std::endl;
    

    
        bool good_base = false; 
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--base") {
                std::ifstream base_file(argv[i + 1], std::ios::binary);
                if (!base_file.is_open()) {
                    printError("No such file " + std::string(argv[i + 1]));
                    return 0;
                }
                base_filename = argv[i + 1];
                base_file.close();
                good_base = true;
                break;
            }
        }
        if (!good_base) {
            printError("Base file was not specified");
            return 0;
        }
    
    
        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--outputGraph") {
                graphname = std::string(argv[i + 1]);
                break;
            }
        }
        if (graphname.empty()) {
            printError("Filename of the output graph was not specified");
            return 0;
        }
        std::cout << "Output graph: " << graphname << std::endl;


        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--relevanceVector") {
                if (sscanf(argv[i + 1], "%d", &relevanceVector) != 1 || relevanceVector <= 0) {
                    printError("Inappropriate value for the relevance vector length: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        if (relevanceVector == -1) {
            printError("Relevance vector length was not specified");
            return 0;
        }
        std::cout << "Relevance vector length: " << relevanceVector << std::endl;
       
 
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--efConstruction") {
                if (sscanf(argv[i + 1], "%d", &efConstruction) != 1 || efConstruction <= 0) {
                    printError("Inappropriate value for efConstruction: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        std::cout << "efConstruction: " << efConstruction << std::endl;
        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--M") {
                if (sscanf(argv[i + 1], "%d", &M) != 1 || M <= 0) {
                    printError("Inappropriate value for M: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        std::cout << "M: " << M << std::endl;
    
    } else if (mode == "query") {
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--querySize") {
                if (sscanf(argv[i + 1], "%d", &qsize) != 1 || qsize <= 0) {
                    printError("Inappropriate value for query size: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        if (qsize == -1) {
            printError("Query size was not specified");
            return 0;
        }
        std::cout << "Query size: " << qsize << std::endl;
        


        bool good_query = false; 
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--query") {
                std::ifstream query_file(argv[i + 1], std::ios::binary);
                if (!query_file.is_open()) {
                    printError("No such file " + std::string(argv[i + 1]));
                    return 0;
                }
                query_file.close();
                query_filename = argv[i + 1];
                good_query = true;
                break;
            }
        }
        if (!good_query) {
            printError("Query file was not specified");
            return 0;
        }
        
        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--inputGraph") {
                graphname = std::string(argv[i + 1]);
                break;
            }
        }
        if (graphname.empty()) {
            printError("Filename of the input graph was not specified");
            return 0;
        }
        std::cout << "Input graph: " << graphname << std::endl;


        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--efSearch") {
                if (sscanf(argv[i + 1], "%d", &efSearch) != 1 || efSearch <= 0) {
                    printError("Inappropriate value for efSearch: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        std::cout << "efSearch: " << efSearch << std::endl;
        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--topK") {
                if (sscanf(argv[i + 1], "%d", &topK) != 1 || topK <= 0) {
                    printError("Inappropriate value for top size: \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                break;
            }
        }
        std::cout << "Top size: " << topK << std::endl;
        
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--output") {
                std::ofstream output(argv[i + 1]);
                if (!output.is_open()) {
                    printError("Cannot open file \"" + std::string(argv[i + 1]) + "\"");
                    return 0;
                }
                output.close();
                outputname = std::string(argv[i + 1]);
                break;
            }
        }
        if (outputname.empty()) {
            std::cout << "Output file: " << "stdout" << std::endl;
        } else {
            std::cout << "Output file: " << outputname << std::endl;
        }




        good_gt = false; 
        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--groundtruth") {
                std::ifstream gt_file(argv[i + 1], std::ios::binary);
                if (!gt_file.is_open()) {
                    printError("No such file " + std::string(argv[i + 1]));
                    return 0;
                }
                gt_file.close();
                gt_filename = argv[i + 1];
                good_gt = true;
                break;
            }
        }

        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--gtQueries") {
                if (sscanf(argv[i + 1], "%d", &gtQueries) != 1 || gtQueries <= 0) {
                    std::cerr << "Inappropriate value for the number of queries: \"" << std::string(argv[i + 1]) << "\". We do not compute recall" << std::endl;
                    good_gt = false;
                }
                break;
            }
        }
        if (gtQueries == -1 && good_gt) {
            printError("Number of queries in groundtruth file was not specified. We do not compute recall");
            good_gt = false;
        }

        for (int i = 1; i < argc - 1; i++) {
            if (std::string(argv[i]) == "--gtTop") {
                if (sscanf(argv[i + 1], "%d", &gtTop) != 1 || gtTop <= 0) {
                    std::cerr << "Inappropriate value for the top size: \"" << std::string(argv[i + 1]) << "\". We do not compute recall" << std::endl;
                    good_gt = false;
                }
                if (gtTop < topK) {
                    std::cerr << "Groundtruth top cannot be less than retrieval top. We do not compute recall" << std::endl;
                    good_gt = false;
                }
                break;
            }
        }
        if (gtTop == -1 && good_gt) {
            printError("Top size in groundtruth file was not specified. We do not compute recall");
            good_gt = false;
        }

    }

        





    if (mode == "base") {
        hnswlib::InitializeBaseConstruction(base_filename, vecsize, trainQueries, relevanceVector);
        hnswlib::L2Space l2space(1);
        float *mass = new float[vecsize];
        for (int i = 0; i < vecsize; i++) {
            mass[i] = (float)i;
        }
        
        appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, vecsize, M, efConstruction);
        std::cout << "Building index" << std::endl;
        double t1 = omp_get_wtime();
        for (int i = 0; i < 1; i++) {
            appr_alg->addPoint((void *)(mass + i), (size_t)i);
        }

        #pragma omp parallel for
        for (int i = 1; i < vecsize; i++) {
            appr_alg->addPoint((void *)(mass + i), (size_t)i);
        }
        double t2 = omp_get_wtime();
 
        std::cout << "Index built, time=" << t2 - t1 << " s" << "\n";
        appr_alg->SaveIndex(graphname.data());
        //delete appr_alg;
        //delete mass;
    } else {

        hnswlib::InitializeSearch(query_filename, vecsize, qsize);
        
        hnswlib::L2Space l2space(1);
        float *massQ = new float[qsize];

        for (int i = 0; i < qsize; i++) {
            massQ[i] = (float)i;
        }

        std::priority_queue< std::pair< float, labeltype >> gt[qsize];
		
        appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, graphname.data());
        appr_alg->setEf(efSearch);
        std::ofstream fres;
        if (!outputname.empty()) {
            fres.open(outputname);
        }
      
        float sum = 0;
 
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < qsize; i++) {
            gt[i] = appr_alg->searchKnn(massQ + i, topK);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        std::vector <std::vector <int> > answer;
        for (int i = 0; i < qsize; i++) {
            std::vector <int> res;
            while (!gt[i].empty()) {
                res.push_back(gt[i].top().second);
                sum += -gt[i].top().first;
                gt[i].pop();
            }
            std::reverse(res.begin(), res.end());
            answer.push_back(res);
            for (auto it: res) {
                if (!outputname.empty()) {
                    fres << it << ' ';
                } else {
                    std::cout << it << ' ';
                }
            }
            if (!outputname.empty()) {
                fres << std::endl;
            } else {
                std::cout << std::endl;
            }
        }
        if (!outputname.empty()) {
            fres.close();
        }
        std::cout << "Average query time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / (double)qsize << "ms" << std::endl;
 
        std::cout << "Average relevance: " << sum / (qsize * topK) << std::endl;
        std::cout << "Average number of model computations: " << hnswlib::geDistanceCounter() / (double)(qsize) << std::endl;
 
        if (good_gt) {

            std::vector <std::vector <int> > gt(gtQueries, std::vector <int>(gtTop));
            std::ifstream gt_file(gt_filename, std::ios::binary);
            for (int i = 0; i < gtQueries; i++) {
                gt_file.read((char*)gt[i].data(), sizeof(gt[0][0]) * gtTop);
            }
            gt_file.close();
            
            int cnt = 0;
            for (int i = 0; i < qsize; i++) {
                std::vector <int> res = answer[i];
                std::vector <int> gt_i;

                for (int j = 0; j < topK; j++) {
                    gt_i.push_back(gt[i][j]);
                }

                std::sort(res.begin(), res.end());
                std::sort(gt_i.begin(), gt_i.end());
                
                std::vector <int> intersection;
                std::set_intersection(res.begin(), res.end(), gt_i.begin(), gt_i.end(), std::back_inserter(intersection));
                cnt += intersection.size();
            }
            std::cout << "Recall@" << topK << ": " << cnt / (double) (qsize * topK) << std::endl; 


        }
    }
    return 0;
}
