// nnet3/nnet-compute.h

// Copyright   2012-2015  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_NNET3_NNET_COMPUTE_H_
#define KALDI_NNET3_NNET_COMPUTE_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-nnet.h"
#include "nnet3/nnet-computation.h"
#include "nnet3/nnet-analyze.h"
#include "nnet3/nnet-example.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <map>


namespace kaldi {
namespace nnet3 {



/**
  class NnetComputer is responsible for executing the computation described in the
  "computation" object.

  You call in sequence, the constructor, then AcceptInput() [or AcceptInputs()],
  then Forward(), then GetOutput(), then if applicable (Backward(), then if
  applicable GetInputDeriv()).
 */
class NnetOnlineComputer {
 public:
  /// Constructor.  nnet_to_update will be NULL if you are not doing
  /// model update or model-derivative computation.
  /// You must call computation.ComputeCudaIndexes()  before calling
  /// this function.
  NnetOnlineComputer(const NnetComputation &computation,
               const Nnet &nnet,bool pad_input);

  /// e.g. AcceptInput ("input", input_mat).  Will crash if there is no
  /// input node with the given name.  This function is destructive of "input"
  /// as it takes it using the Swap function of CuMatrix.
  /// Must have the same number of rows as the corresponding input described
  /// in the ComputationRequest e.g. the indexes.size() in the corresponding
  /// IoSpecification.
  void AcceptInput(const std::string &input_name,
                   CuMatrix<BaseFloat> *input);

  /// This function calls AcceptInput() in turn on all the inputs in the
  /// training example (provide example.io; this interface makes it easy to work
  /// with CCTC examples too).  It needs "nnet" only in order to distinguish
  /// inputs from outputs.
  void AcceptInputs(const Nnet &nnet,
                    const std::vector<NnetIo> &io);


  // Does the forward computation.
  void Forward();

  // e.g. GetOutput ("output").  Will crash if no such output.
  const CuMatrixBase<BaseFloat> &GetOutput(const std::string &output_name) const;

  // This function works as follows: given a chunk of input (interpreted
  // as following in time any previously supplied data), do the computation
  // and produce all the frames of output we can.  In the middle of the
  // file, the dimensions of input and output will be the same, but at
  // the beginning of the file, output will have fewer frames than input
  // due to required context.
  // It is the responsibility of the user to keep track of frame indices, if
  // required.  This class won't output any frame twice.
  void Compute(const CuMatrixBase<BaseFloat> &input,
               CuMatrix<BaseFloat> *output);
  
  // This flushes out the last frames of output; you call this when all
  // input has finished.  It's invalid to call Compute or Flush after
  // calling Flush.  It's valid to call Flush if no frames have been
  // input or if no frames have been output; this produces empty output.
  void Flush(CuMatrix<BaseFloat> *output);

 private:  
  const NnetComputation &computation_;
  const Nnet &nnet_;

  // command_attributes_ is only used if debug_=true.
  std::vector<CommandAttributes> command_attributes_;
  // submatrix_strings_ is only used if debug_=true.
  std::vector<std::string> submatrix_strings_;
  // command_strings_ is only used if debug_=true, or in case of error.
  std::vector<std::string> command_strings_;

  // The matrices used in the computation.
  std::vector<CuMatrix<BaseFloat> > matrices_;

  bool pad_input_;  // pad input at the beginning of the decode

  // executes the command in computation_.commands[command].
  void ExecuteCommand(int32 command);

  // Returns the matrix index where the input or output matrix index for
  // "node_name" is stored (or its corresponding derivative, if is_deriv==true).
  // "is_output" tells the code that this is an output node, as opposed to an
  // input node; it's used only for checking.
  int32 GetMatrixIndex(const std::string &node_name,
                       bool is_output, bool is_deriv) const;

  CuSubMatrix<BaseFloat> GetSubMatrix(int32 submatrix_index);

  void GetPointers(int32 indexes_multi_index,
                   int32 num_cols,
                   CuArray<BaseFloat*> *pointers);
  void GetPointers(int32 indexes_multi_index,
                   int32 num_cols,
                   CuArray<const BaseFloat*> *pointers);

};



} // namespace nnet3
} // namespace kaldi

#endif
