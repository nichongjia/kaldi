// thread/kaldi-semaphore.h

// Copyright 2012  Karel Vesely (Brno University of Technology)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#ifndef KALDI_THREAD_KALDI_SEMAPHORE_H_
#define KALDI_THREAD_KALDI_SEMAPHORE_H_ 1

#include <pthread.h>

namespace kaldi {
  
class Semaphore {
 public:
  Semaphore(int32 initValue = 0); 
  ~Semaphore();

  bool TryWait(); ///< Returns true if Wait() goes through
  void Wait(); ///< decrease the counter
  void Signal(); ///< increase the counter
  
  /**
   * returns the counter value, 
   * zero means no resources, the Wait() will block
   */ 
  int32 GetValue() {
    return counter_; 
  }

 private:
  int32 counter_; ///< the semaphore counter, 0 means block on Wait() 
  
  pthread_mutex_t mutex_;
  pthread_cond_t cond_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(Semaphore);
};



} //namespace

#endif // KALDI_THREAD_KALDI_SEMAPHORE_H_
