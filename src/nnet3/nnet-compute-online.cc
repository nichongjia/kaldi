// nnet3/nnet-compute.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

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

#include <iterator>
#include <sstream>
#include "nnet3/nnet-compute-online.h"

namespace kaldi {
namespace nnet3 {


NnetOnlineComputer::NnetOnlineComputer(const NnetComputation &computation,
               const Nnet &nnet,bool pad_input):
    computation_(computation), nnet_(nnet),pad_input_(pad_input) {
  KALDI_ASSERT(computation.indexes_cuda.size() == computation.indexes.size() &&
 computation.indexes_ranges_cuda.size() == computation.indexes_ranges.size() &&
               "You must call NnetComputation::ComputeCudaIndexes() before "
               "executing the computation.");
  matrices_.resize(computation.matrices.size());  
}

void NnetOnlineComputer::ExecuteCommand(int32 command) {
  const NnetComputation::Command &c = computation_.commands[command];
  try {
    switch (c.command_type) {
      case kAllocMatrixZeroed:
        matrices_[c.arg1].Resize(computation_.matrices[c.arg1].num_rows,
                                 computation_.matrices[c.arg1].num_cols,
                                 kSetZero,
                                 computation_.matrices[c.arg1].stride_type);
        break;
      case kAllocMatrixUndefined:
        matrices_[c.arg1].Resize(computation_.matrices[c.arg1].num_rows,
                                 computation_.matrices[c.arg1].num_cols,
                                 kUndefined,
                                 computation_.matrices[c.arg1].stride_type);
        break;
      case kDeallocMatrix:
        matrices_[c.arg1].Resize(0, 0);
        break;
      case kAllocMatrixFromOther:
        matrices_[c.arg1].Swap(&(matrices_[c.arg2]));
        break;
      case kAllocMatrixFromOtherZeroed:
        matrices_[c.arg1].Swap(&(matrices_[c.arg2]));
        matrices_[c.arg1].SetZero();
        break;
      case kPropagate: {
        const Component *component = nnet_.GetComponent(c.arg1);
        ComponentPrecomputedIndexes *indexes =
            computation_.component_precomputed_indexes[c.arg2];
        const CuSubMatrix<BaseFloat> input(GetSubMatrix(c.arg3));
        CuSubMatrix<BaseFloat> output(GetSubMatrix(c.arg4));
        component->Propagate(indexes, input, &output);
        break;
      }
      case kStoreStats: {
        KALDI_ASSERT(nnet_to_update_ != NULL);
        Component *upd_component = nnet_to_update_->GetComponent(c.arg1);
        CuSubMatrix<BaseFloat> output(GetSubMatrix(c.arg2));
        upd_component->StoreStats(output);
        break;
      }
      case kBackprop:
      case kBackpropNoModelUpdate:  {
        std::ostringstream debug_str;
        KALDI_ASSERT(nnet_to_update_ != NULL);
        debug_str << nnet_.GetComponentName(c.arg1);
        const Component *component = nnet_.GetComponent(c.arg1);
        KALDI_ASSERT(!(computation_.need_model_derivative && !nnet_to_update_));
        Component *upd_component = (nnet_to_update_ &&
                                    c.command_type == kBackprop &&
                                    computation_.need_model_derivative ?
                                    nnet_to_update_->GetComponent(c.arg1) :
                                    NULL);
        ComponentPrecomputedIndexes *indexes =
            computation_.component_precomputed_indexes[c.arg2];
        const CuSubMatrix<BaseFloat> in_value(GetSubMatrix(c.arg3));
        const CuSubMatrix<BaseFloat> out_value(GetSubMatrix(c.arg4));
        const CuSubMatrix<BaseFloat> out_deriv(GetSubMatrix(c.arg5));
        CuSubMatrix<BaseFloat> in_deriv(GetSubMatrix(c.arg6));
        component->Backprop(debug_str.str(), indexes,
                            in_value, out_value, out_deriv, upd_component,
                            c.arg6 == 0 ? NULL : &in_deriv);
        break;
      }
      case kMatrixCopy: {
        CuSubMatrix<BaseFloat> dest(GetSubMatrix(c.arg1));
        const CuSubMatrix<BaseFloat> src(GetSubMatrix(c.arg2));
        dest.CopyFromMat(src);
        break;
      }
      case kMatrixAdd: {
        CuSubMatrix<BaseFloat> dest(GetSubMatrix(c.arg1));
        const CuSubMatrix<BaseFloat> src(GetSubMatrix(c.arg2));
        dest.AddMat(1.0, src);
        break;
      }
      case kAddRows: {
        CuSubMatrix<BaseFloat> dest(GetSubMatrix(c.arg1));
        const CuSubMatrix<BaseFloat> src(GetSubMatrix(c.arg2));
        const CuArray<int32> &indexes = computation_.indexes_cuda[c.arg3];
        dest.AddRows(1.0, src, indexes);
        break;
      }
      case kCopyRows: {
        CuSubMatrix<BaseFloat> dest(GetSubMatrix(c.arg1));
        const CuSubMatrix<BaseFloat> src(GetSubMatrix(c.arg2));
        const CuArray<int32> &indexes = computation_.indexes_cuda[c.arg3];
        dest.CopyRows(src, indexes);
        break;
      }
      case kCopyRowsMulti: {
        CuSubMatrix<BaseFloat> dest(GetSubMatrix(c.arg1));
        CuArray<const BaseFloat*> pointers;
        GetPointers(c.arg2, dest.NumCols(), &pointers);
        dest.CopyRows(pointers);
        break;
      }
      case kCopyToRowsMulti: {
        CuSubMatrix<BaseFloat> src(GetSubMatrix(c.arg1));
        CuArray<BaseFloat*> pointers;
        GetPointers(c.arg2, src.NumCols(), &pointers);
        src.CopyToRows(pointers);
        break;
      }
      case kAddRowsMulti: {
        CuSubMatrix<BaseFloat> dest(GetSubMatrix(c.arg1));
        CuArray<const BaseFloat*> pointers;
        GetPointers(c.arg2, dest.NumCols(), &pointers);
        dest.AddRows(1.0, pointers);
        break;
      }
      case kAddToRowsMulti: {
        CuSubMatrix<BaseFloat> src(GetSubMatrix(c.arg1));
        CuArray<BaseFloat*> pointers;
        GetPointers(c.arg2, src.NumCols(), &pointers);
        src.AddToRows(1.0, pointers);
        break;
      }
      case kAddRowRanges: {
        CuSubMatrix<BaseFloat> dest(GetSubMatrix(c.arg1));
        const CuSubMatrix<BaseFloat> src(GetSubMatrix(c.arg2));
        const CuArray<Int32Pair> &pairs = computation_.indexes_ranges_cuda[c.arg3];
        dest.AddRowRanges(src, pairs);
        break;
      }
      case kNoOperation: case kNoOperationMarker:
        break;
      default:
        KALDI_ERR << "Invalid command in computation";
    }
  } catch (...) {
    if (!debug_) {
      std::string preamble;
      computation_.GetCommandStrings(nnet_, &preamble, &command_strings_);
      KALDI_WARN << "Printing some background info since error was detected";
      KALDI_LOG << preamble;
      for (int32 prev_c = 0; prev_c < command; prev_c++)
        KALDI_LOG << command_strings_[prev_c];
    }
    // the following will re-throw the error, but now we've printed more info
    // about what went wrong.
    KALDI_ERR << "Error running command " << command_strings_[command];
  }
}

CuSubMatrix<BaseFloat> NnetOnlineComputer::GetSubMatrix(int32 submatrix_index) {
  KALDI_PARANOID_ASSERT(static_cast<size_t>(submatrix_index) <
                        computation_.submatrices.size());
  const NnetComputation::SubMatrixInfo &info =
      computation_.submatrices[submatrix_index];
  const CuMatrix<BaseFloat> &mat = matrices_[info.matrix_index];
  return CuSubMatrix<BaseFloat>(
      mat, info.row_offset, info.num_rows, info.col_offset, info.num_cols);
}

void NnetOnlineComputer::GetPointers(int32 indexes_multi_index,
                               int32 num_cols,
                               CuArray<BaseFloat*> *pointers) {
  KALDI_ASSERT(static_cast<size_t>(indexes_multi_index)
               < computation_.indexes_multi.size());
  const std::vector<std::pair<int32,int32> > &pairs =
      computation_.indexes_multi[indexes_multi_index];
  int32 size = pairs.size();
  std::vector<BaseFloat*> vec(size);

  // the map "lookup" maps from submatrix index to the Data()
  // pointer of that submatrix, and the corresponding Stride().
  unordered_map<int32, std::pair<BaseFloat*, int32> > lookup;

  for (int32 i = 0; i < size; i++) {
    int32 submatrix_index = pairs[i].first, row = pairs[i].second;
    if (submatrix_index != -1) {
      unordered_map<int32, std::pair<BaseFloat*, int32> >::iterator
          iter = lookup.find(submatrix_index);
      if (iter == lookup.end()) {
        CuSubMatrix<BaseFloat> m = GetSubMatrix(submatrix_index);
        lookup[submatrix_index] = std::pair<BaseFloat*, int32>(m.Data(),
                                                               m.Stride());
        iter = lookup.find(submatrix_index);
      }
      BaseFloat *data = iter->second.first;
      int32 stride = iter->second.second;
      vec[i] = data + (row * stride);
    } else {
      // -1 is a marker that will be translated to NULL.
      vec[i] = NULL;
    }
  }
#ifdef KALDI_PARANOID
  for (int32 i = 0; i < size; i += 30 + RandInt(0, 9)) {
    // Do a pseudo-random spot check that the row-indexes are not out of range.
    int32 submatrix_index = pairs[i].first, row = pairs[i].second;
    CuSubMatrix<BaseFloat> m = GetSubMatrix(submatrix_index);
    KALDI_ASSERT(row >= 0 && row < m.NumRows() && num_cols == m.NumCols());
  }
#endif
  pointers->CopyFromVec(vec);
}

void NnetOnlineComputer::GetPointers(int32 indexes_multi_index,
                               int32 num_cols,
                               CuArray<const BaseFloat*> *pointers) {
  GetPointers(indexes_multi_index, num_cols,
              reinterpret_cast<CuArray<BaseFloat*>*>(pointers));
}

void NnetOnlineComputer::Forward() {
  CheckInputs(false);
  int32 size = computation_.commands.size(), i = 0;
  const std::vector<NnetComputation::Command> &c = computation_.commands;
  CommandDebugInfo info;
  Timer timer;
  double total_elapsed_previous = 0.0;

  for (; i < size && c[i].command_type != kNoOperationMarker;
       i++) {
    if (debug_)
      DebugBeforeExecute(i, &info);
    ExecuteCommand(i);
    if (debug_) {
      double total_elapsed_now = timer.Elapsed();
      DebugAfterExecute(i, info, total_elapsed_now - total_elapsed_previous);
      total_elapsed_previous = total_elapsed_now;
    }

  }

}

void NnetOnlineComputer::AcceptInput(const std::string &input_name,
                               CuMatrix<BaseFloat> *input) {
  bool is_output = false, is_deriv = false;
  int32 matrix_index = GetMatrixIndex(input_name, is_output, is_deriv);
  KALDI_ASSERT(static_cast<size_t>(matrix_index) < matrices_.size());
  const NnetComputation::MatrixInfo &matrix_info =
      computation_.matrices[matrix_index];
  if (input->NumRows() != matrix_info.num_rows)
    KALDI_ERR << "Num-rows mismatch for input '" << input_name
              << "': " << matrix_info.num_rows
              <<  " in computation-request, " << input->NumRows()
              << " provided.";
  if (input->NumCols() != matrix_info.num_cols)
    KALDI_ERR << "Num-cols mismatch for input '" << input_name
              << "': " << matrix_info.num_cols
              <<  " in computation-request, " << input->NumCols()
              << " provided.";
  if (matrix_info.stride_type == kDefaultStride ||
      input->Stride() == input->NumCols()) {
    matrices_[matrix_index].Swap(input);
  } else {
    matrices_[matrix_index].Resize(matrix_info.num_rows,
                                   matrix_info.num_cols,
                                   kUndefined, kStrideEqualNumCols);
    matrices_[matrix_index].CopyFromMat(*input);
  }
  input->Resize(0, 0);
}

const CuMatrixBase<BaseFloat> &NnetOnlineComputer::GetOutput(
    const std::string &output_name) const {
  bool is_output = true, is_deriv = false;
  int32 matrix_index = GetMatrixIndex(output_name, is_output, is_deriv);
  if (matrices_[matrix_index].NumRows() == 0)
    KALDI_ERR << "GetOutput called when output not ready (before Forward()?)";
  return matrices_[matrix_index];
}

int32 NnetOnlineComputer::GetMatrixIndex(
    const std::string &node_name, bool is_output, bool is_deriv) const {
  int32 node_index = nnet_.GetNodeIndex(node_name);
  if (node_index == -1)
    KALDI_ERR << "No node named '" << node_name << "'in network.";
  if (is_output) {
    if (!nnet_.IsOutputNode(node_index))
      KALDI_ERR << "Expecting output node; node named '"
                << node_name  << "' is not output node.";
  } else {
    if (nnet_.IsOutputNode(node_index))
      KALDI_ERR << "Expecting input node or component node; node named '"
                << node_name  << "' is output node.";
  }
  unordered_map<int32, std::pair<int32, int32> >::const_iterator
      iter = computation_.input_output_info.find(node_index),
      end = computation_.input_output_info.end();
  if (iter == end)
    KALDI_ERR << "Not expecting input or output for node named '" << node_name
              << "' (not in computation request)";
  std::pair<int32,int32> locations = iter->second;
  int32 location;
  if (is_deriv) {
    location = locations.second;
    if (locations.second <= 0) // No deriv expected.
      KALDI_ERR << "Not expecting derivative information for node named '"
                << node_name << "' (not in computation request)";
  } else {
    location = locations.first;
  }
  KALDI_ASSERT(static_cast<size_t>(location) < matrices_.size());
  return location;
}

void NnetOnlineComputer::CheckInputs(bool check_output_deriv) const {
  unordered_map<int32, std::pair<int32, int32> >::const_iterator
      iter = computation_.input_output_info.begin(),
      end = computation_.input_output_info.end();
  for (; iter != end; ++iter) {
    int32 node_index = iter->first,
      value_matrix_index = iter->second.first,
      deriv_matrix_index = iter->second.second;
    std::string name = nnet_.GetNodeName(node_index);
    if (nnet_.IsOutputNode(node_index)) {
      if (check_output_deriv && deriv_matrix_index > 0) {
        KALDI_ASSERT(static_cast<size_t>(deriv_matrix_index) < matrices_.size());
        if (matrices_[deriv_matrix_index].NumRows() == 0)
          KALDI_ERR << "Output-derivative required but not provided for node '"
                    << name << "'.";
      }
    } else {
      if (!check_output_deriv) {
        if (matrices_[value_matrix_index].NumRows() == 0)
          KALDI_ERR << "Input required but not provided for node '"
                    << name << "'.";
      }
    }
  }
}

void NnetOnlineComputer::AcceptInputs(const Nnet &nnet,
                                const std::vector<NnetIo> &io_vec) {
  for (size_t i = 0; i < io_vec.size(); i++) {
    const NnetIo &io = io_vec[i];
    int32 node_index = nnet.GetNodeIndex(io.name);
    if (node_index == -1)
      KALDI_ERR << "No node named '" << io.name << "' in nnet.";
    if (nnet.IsInputNode(node_index)) {
      CuMatrix<BaseFloat> cu_input(io.features.NumRows(),
                                   io.features.NumCols(),
                                   kUndefined);
      cu_input.CopyFromGeneralMat(io.features);
      this->AcceptInput(io.name, &cu_input);
    }
  }
}

void NnetOnlineComputer::Compute(const CuMatrixBase<BaseFloat> &input,
               CuMatrix<BaseFloat> *output) {
}

void Flush(CuMatrix<BaseFloat> *output) {

}

} // namespace nnet3
} // namespace kaldi
