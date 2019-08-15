#pragma once
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define  __builtin_popcount(t) __popcnt(t)

#endif
typedef size_t labeltype;
typedef unsigned int tableint;
#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif

namespace hnswlib {
	template<typename MTYPE>
	using DISTFUNC = MTYPE(*) (const void *, const void *);



	template <typename dist_t>  class AlgorithmInterface {
	public:
		virtual std::priority_queue< std::pair< dist_t, labeltype >> searchKnn(void *,int) = 0;
	};
	template<typename MTYPE>
	class SpaceInterface {
	public:
		virtual DISTFUNC<MTYPE> get_dist_func() = 0;
		virtual void *get_dist_func_param() = 0;

	};
}
#include "L2space.h"
#include "hnswalg.h"
