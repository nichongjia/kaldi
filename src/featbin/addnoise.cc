// featbin/addnoise.cc

// Copyright 2009-2012  Microsoft Corporation
//                      Johns Hopkins University (author: Daniel Povey)

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include <cmath>
#include <strstream>

int main(int argc, char *argv[]) {
	try {
		using namespace kaldi;
		const char *usage =
			"Add noise into a file from a source noise list to randomly select. "
			"Usage:  compute-mfcc-feats [options...] <wav-rspecifier> <noise-wav-respefier> snr-low snr-high <dstdir>\n";

		// construct all the global objects
		ParseOptions po(usage);

		
		po.Read(argc, argv);

		if (po.NumArgs() != 5) {
			po.PrintUsage();
			exit(1);
		}

		std::string wav_rspecifier = po.GetArg(1);
		std::string noise_wav_rspecifier = po.GetArg(2);		
		std::string snr_low = po.GetArg(3);
		std::string snr_high = po.GetArg(4);
		std::string dst_dir = po.GetArg(5);

		SequentialTableReader<WaveHolder> wav_reader(wav_rspecifier);
		SequentialTableReader<WaveHolder> noise_wav_reader(noise_wav_rspecifier);
		

		int32 nsnr_low = atoi(snr_low.c_str());
		int32 nsnr_high = atoi(snr_high.c_str());
		
		int32 nnum_noise_wavs = 0, m = 0, n_data_samples, n_noise_samples;
		BaseFloat fwav_energy = 0.0, fnoise_engery = 0.0;

		std::vector<std::string> vec_noise_name;
		std::vector<WaveData> vec_noise_wav;
		std::vector<BaseFloat> vec_noise_energy;


		for (; !noise_wav_reader.Done(); noise_wav_reader.Next()) {
			std::string utt = noise_wav_reader.Key();
			const WaveData &wave_data = noise_wav_reader.Value();
			vec_noise_name.push_back(utt);
			vec_noise_wav.push_back(wave_data);
		}

		nnum_noise_wavs = vec_noise_wav.size();
		srand(0);

		for (; !wav_reader.Done(); wav_reader.Next()) {
			std::string utt = wav_reader.Key();
			const WaveData &wave_data = wav_reader.Value();
			const Matrix<BaseFloat> data_matrix = wave_data.Data();
			int32 nnoise_ind = rand() % nnum_noise_wavs;
			const Matrix<BaseFloat> noise_data_matrix = vec_noise_wav[nnoise_ind].Data();
			n_noise_samples = noise_data_matrix.NumCols();
			n_data_samples = data_matrix.NumCols();

			//compute the data energy
			fwav_energy = 0.0;
			for (m = 0; m < n_data_samples; m++){
				fwav_energy += data_matrix(0, m)*data_matrix(0, m);		   
			}

			// Rand the start position 
			int32 n_noise_beg = rand() % n_noise_samples;			
			// make a noise copy, and compute the noise energy
			fnoise_engery = 0.0;
			Matrix<BaseFloat> noise_copy(data_matrix.NumRows(), data_matrix.NumCols());

			std::vector<BaseFloat> vec_noise_copy(n_data_samples);
			for (m = 0; m < n_data_samples; m++){
				int32 nindex = (n_noise_beg + m) % n_noise_samples;
				noise_copy(0, m) = noise_data_matrix(0, nindex);				
				fnoise_engery += noise_copy(0, m) * noise_copy(0, m);
			}

			KALDI_ASSERT(nsnr_high > nsnr_low);
			int nsnr = rand() % (nsnr_high - nsnr_low + 1) + nsnr_low;
			std::strstream ss;
			std::string str_SNR;
			ss << nsnr;
			ss >> str_SNR;
				
			BaseFloat tmp_f = nsnr / 10.0;
			BaseFloat tmp_fp = pow(10.0, tmp_f);
			BaseFloat k_f = sqrt(fwav_energy / (fnoise_engery*tmp_fp));

			
			Matrix<BaseFloat> add_noised_data(data_matrix);
			add_noised_data.AddMat(k_f, noise_copy);
			
			// convert the data into int16
			for(m = 0; m < add_noised_data.NumCols(); m++) {
				add_noised_data(0,m) = static_cast<int16>(add_noised_data(0,m));
			}
			WaveData add_noised_wav(wave_data.SampFreq(), add_noised_data);

			std::string strDstWavFn;
			std::string noise_key;
			noise_key = vec_noise_name[nnoise_ind];
			strDstWavFn = dst_dir + "//" + utt + "_" + noise_key + "_" + str_SNR + ".wav";
			std::ofstream file(strDstWavFn,std::ofstream::binary);
			add_noised_wav.Write(file);		
		}		
	}
	catch (const std::exception &e) {
		std::cerr << e.what();
		return -1;
	}
}

