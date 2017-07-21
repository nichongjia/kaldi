
all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += -lfcgi -lfcgi++ $(CUDA_LDLIBS)
EXTRA_CXXFLAGS += -I$(KALDI_PATH) -L$(KALDI_PATH) $(APIAI_CXX_FLAGS)

OBJFILES = Timing.o Response.o RequestRawReader.o ResponseJsonWriter.o ResponseMultipartJsonWriter.o OnlineDecoder.o Nnet3LatgenFasterDecoder.o QueryStringParser.o FcgiDecodingApp.o 

LIBNAME = libstidecoder

BINFILES = fcgi-nnet3-decoder

TESTFILES = QueryStringParserTests

ADDLIBS = ../online2/kaldi-online2.a ../ivector/kaldi-ivector.a \
          ../nnet2/kaldi-nnet2.a ../nnet3/kaldi-nnet3.a ../lat/kaldi-lat.a \
          ../decoder/kaldi-decoder.a  ../cudamatrix/kaldi-cudamatrix.a \
          ../feat/kaldi-feat.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../thread/kaldi-thread.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a \
          ../matrix/kaldi-matrix.a ../fstext/kaldi-fstext.a \
          ../util/kaldi-util.a ../base/kaldi-base.a 
          
include ../makefiles/default_rules.mk
