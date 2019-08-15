#pragma once
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define  __builtin_popcount(t) __popcnt(t)
#else
#include <x86intrin.h>
#endif
#define USE_AVX
#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif

#include <map>
#include <cmath>
#include <cassert>
#include <set>
#include <string>

#include <omp.h>

#include <ctime>


int distance_counter = 0;
std::vector <std::vector <float> > model_features_train;
std::vector <std::vector <float> > model_features_test;
int relevanceVectorLength = -1;


#include "hnswlib.h"
namespace hnswlib {
    using namespace std;

    void InitializeBaseConstruction(std::string basefile_, int baseSize_, int trainSize_, int relevanceVectorLength_)
    {
        model_features_train = std::vector <std::vector <float> >(baseSize_, std::vector <float>(trainSize_));
        
        std::ifstream train_features(basefile_, std::ios::binary);
        for (int i = 0; i < baseSize_; i++) {
            train_features.read((char*)model_features_train[i].data(), sizeof(model_features_train[0][0]) * trainSize_);
        }
        train_features.close();

        relevanceVectorLength = relevanceVectorLength_; 
    }

    void InitializeSearch(std::string queryfile_, int baseSize_, int querySize_)
    {
        model_features_test = std::vector <std::vector <float> >(baseSize_, std::vector <float>(querySize_));
        
        std::ifstream test_features(queryfile_, std::ios::binary);
        for (int i = 0; i < baseSize_; i++) {
            test_features.read((char*)model_features_test[i].data(), sizeof(model_features_test[0][0]) * querySize_);
        }
        test_features.close();
    }

	static float
		constructionDistance(const void *pVect1, const void *pVect2)
	{
        float float_query = ((float*)pVect1)[0];
        int idx_query = float_query;
        
        float float_item = ((float*)pVect2)[0];
        int idx_item = float_item;

        float val = 0;
        for (int i = 0; i < relevanceVectorLength; i++) {
            float tmp = model_features_train[idx_item][i] - model_features_train[idx_query][i];
            val += tmp * tmp;
        }

        return val;
	}
	
    static float
		searchDistance(const void *pVect1, const void *pVect2)
	{
        float float_query = ((float*)pVect1)[0];
        int idx_query = float_query;
        
        float float_item = ((float*)pVect2)[0];
        int idx_item = float_item;
        
        distance_counter++;
        return -model_features_test[idx_item][idx_query];
	}

    int geDistanceCounter()
    {
        return distance_counter;
    }
	
	class L2Space : public SpaceInterface<float> {
		
		DISTFUNC<float> fstdistfunc_;
		size_t data_size_;
		size_t dim_;
	public:
		L2Space(size_t dim) {
			fstdistfunc_ = searchDistance;
            dim_ = 1;
		}

		DISTFUNC<float> get_dist_func() {
			return fstdistfunc_;
		}
		void *get_dist_func_param() {
			return &dim_;
		}

    };

}
