#pragma once

#include <stdlib.h>

#include <iterator>

const unsigned int DEFAULT_SEED = 13;

namespace unittest
{

template<typename T>
  struct random_integer
{
  T operator()(void) const
  {
      T value = 0;
        
      for(int i = 0; i < sizeof(T); i++)
          value ^= T(rand() & 0xff) << (8*i);

      return value;
  }
};

template<>
  struct random_integer<bool>
{
  bool operator()(void) const
  {
    return rand() > RAND_MAX/2 ? false : true;
  }
};

template<>
  struct random_integer<float>
{
  float operator()(void) const
  {
      return rand();
  }
};

template<>
  struct random_integer<double>
{
  double operator()(void) const
  {
      return rand();
  }
};


template<typename T>
  struct random_sample
{
  T operator()(void) const
  {
    random_integer<T> rnd;
    return rnd() % 21 - 10;
  } 
}; 

template<>
  struct random_sample<float>
{
  float operator()(void) const
  {
    return 20.0f * (rand() / (RAND_MAX + 1.0f)) - 10.0f;
  }
};

template<>
  struct random_sample<double>
{
  double operator()(void) const
  {
    return 20.0 * (rand() / (RAND_MAX + 1.0)) - 10.0;
  }
};



template<typename ForwardIterator>
void random_integers(ForwardIterator begin, ForwardIterator end, int seed = DEFAULT_SEED)
{
    const size_t N = end - begin;

    srand(DEFAULT_SEED);

    typedef typename std::iterator_traits<ForwardIterator>::value_type T;

    random_integer<T> rnd;

    while(begin != end)
        *begin++ = rnd();
}

template<typename ForwardIterator>
void random_samples(ForwardIterator begin, ForwardIterator end, int seed = DEFAULT_SEED)
{
    const size_t N = end - begin;

    srand(DEFAULT_SEED);

    typedef typename std::iterator_traits<ForwardIterator>::value_type T;

    random_sample<T> rnd;

    while(begin != end)
        *begin++ = rnd();
}

}; //end namespace unittest

