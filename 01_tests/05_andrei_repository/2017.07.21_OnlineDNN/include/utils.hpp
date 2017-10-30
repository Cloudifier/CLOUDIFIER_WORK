#include <stdio.h>
#include <vector>
#include <tuple>

#include "range.hpp"

#define PBSTR "============================================================"
#define PBWIDTH 60

template<typename T1, typename T2>
std::vector< std::tuple <T1, T2> > zip(std::vector<T1> arr1, std::vector<T2> arr2)
{
	std::vector< std::tuple <T1, T2> > zip_arr;

	if (arr1.size() < arr2.size())
		for(auto i : util::lang::indices(arr1))
			zip_arr.push_back(std::make_tuple(arr1[i], arr2[i]));
	else
		for(auto i : util::lang::indices(arr2))
			zip_arr.push_back(std::make_tuple(arr1[i], arr2[i]));

	return zip_arr;
}

template<typename T>
std::vector<T> range(T min, T max, int step = 1)
{
	std::vector<T> result;

	for (T i = min ; i <= max ; i += step)
		result.push_back(i);

	return result;
}

void print_progress(double percentage)
{
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush(stdout);
}
