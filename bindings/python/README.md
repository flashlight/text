# Flashlight Text Python Bindings
### Quickstart

The Flashlight Text Python package containing beam search decoder and Dictionary components is available on PyPI:
```bash
pip install flashlight-text
```
To enable optional KenLM support in Python with the decoder, KenLM must be installed via pip:
```bash
pip install git+https://github.com/kpu/kenlm.git
```

#### Contents
- [Installation](#installation)
  * [Dependencies](#dependencies)
  * [Build Instructions](#build-instructions)
  * [Advanced Options](#advanced-options)
- [Python API Documentation](#python-api-documentation)
  * [Beam search decoder](#beam-search-decoder)
  * [Beam search decoding with your own language model](#decoding-with-your-own-language-model)
- [Examples](#examples)

## Installation
### Dependencies
We require `python >= 3.6` with the following packages installed:
- [cmake](https://cmake.org/) >= 3.18, and `make` (installable via `pip install cmake`)
- [KenLM](https://github.com/kpu/kenlm) (must be installed `pip install git+https://github.com/kpu/kenlm.git`)

### Build Instructions

Once the dependencies are satisfied, from the project root, use:
```
pip install .
```

Using the environment variable `USE_KENLM=0` removes the KenLM dependency but precludes using the decoder with a language model unless you write C++/`pybind11` bindings for your own language model.

Install in editable mode for development:
```
pip install -e .
```

(`pypi` installation coming soon)

**Note:** if you encounter errors, you'll probably have to `rm -rf build dist` before retrying the install.

## Python API Documentation

### Beam Search Decoder
Bindings for the lexicon and lexicon-free beam search decoders are supported for CTC/ASG models only (no seq2seq model support). Out-of-the-box language model support includes KenLM; users can define custom a language model in Python and use it for decoding; see the [documentation](#define-your-own-language-model-for-beam-search-decoding) below.

To run decoder one first should define options:
```python
    from flashlight.lib.text.decoder import LexiconDecoderOptions, LexiconFreeDecoderOptions

    # for lexicon-based decoder
    options = LexiconDecoderOptions(
        beam_size, # number of top hypothesis to preserve at each decoding step
        token_beam_size, # restrict number of tokens by top am scores (if you have a huge token set)
        beam_threshold, # preserve a hypothesis only if its score is not far away from the current best hypothesis score
        lm_weight, # language model weight for LM score
        word_score, # score for words appearance in the transcription
        unk_score, # score for unknown word appearance in the transcription
        sil_score, # score for silence appearance in the transcription
        log_add, # the way how to combine scores during hypotheses merging (log add operation, max)
        criterion_type # supports only CriterionType.ASG or CriterionType.CTC
    )
    # for lexicon free-based decoder
    options = LexiconFreeDecoderOptions(
        beam_size, # number of top hypothesis to preserve at each decoding step
        token_beam_size, # restrict number of tokens by top am scores (if you have a huge token set)
        beam_threshold, # preserve a hypothesis only if its score is not far away from the current best hypothesis score
        lm_weight, # language model weight for LM score
        sil_score, # score for silence appearance in the transcription
        log_add, # the way how to combine scores during hypotheses merging (log add operation, max)
        criterion_type # supports only CriterionType.ASG or CriterionType.CTC
    )
```

Now, prepare a tokens dictionary (tokens for which a model
returns probability for each frame) and a lexicon (mapping between words and their spellings within a tokens set).

For further details on tokens and lexicon file formats, see the [Data Preparation](https://github.com/flashlight/flashlight/tree/master/flashlight/app/asr#data-preparation) documentation in [Flashlight](https://github.com/flashlight/flashlight).

```python
from flashlight.lib.text.dictionary import Dictionary, load_words, create_word_dict

tokens_dict = Dictionary("path/tokens.txt")
# for ASG add used repetition symbols, for example
# token_dict.add_entry("1")
# token_dict.add_entry("2")

lexicon = load_words("path/lexicon.txt") # returns LexiconMap
word_dict = create_word_dict(lexicon) # returns Dictionary
```

To create a KenLM language model, use:

```python
from flashlight.lib.text.decoder import KenLM
lm = KenLM("path/lm.arpa", word_dict) # or "path/lm.bin"
```

Get the unknown and silence token indices from the token and word dictionaries to pass to the decoder:

```python
sil_idx = token_dict.get_index("|")
unk_idx = word_dict.get_index("<unk>")
```

Now, define the lexicon `Trie` to restrict the beam search decoder search:

```python
from flashlight.lib.text.decoder import Trie, SmearingMode
from flashlight.lib.text.dictionary import pack_replabels

trie = Trie(token_dict.index_size(), sil_idx)
start_state = lm.start(False)

def tkn_to_idx(spelling: list, token_dict : Dictionary, maxReps : int = 0):
    result = []
    for token in spelling:
        result.append(token_dict.get_index(token))
    return pack_replabels(result, token_dict, maxReps)


for word, spellings in lexicon.items():
    usr_idx = word_dict.get_index(word)
    _, score = lm.score(start_state, usr_idx)
    for spelling in spellings:
        # convert spelling string into vector of indices
        spelling_idxs = tkn_to_idx(spelling, token_dict, 1)
        trie.insert(spelling_idxs, usr_idx, score)

    trie.smear(SmearingMode.MAX) # propagate word score to each spelling node to have some lm proxy score in each node.
```

Finally, we can run lexicon-based decoder:

```python
import numpy
from flashlight.lib.text.decoder import LexiconDecoder


blank_idx = token_dict.get_index("#") # for CTC
transitions = numpy.zeros((token_dict.index_size(), token_dict.index_size()) # for ASG fill up with correct values
is_token_lm = False # we use word-level LM
decoder = LexiconDecoder(options, trie, lm, sil_idx, blank_idx, unk_idx, transitions, is_token_lm)
# emissions is numpy.array of emitting model predictions with shape [T, N], where T is time, N is number of tokens
results = decoder.decode(emissions.ctypes.data, T, N)
# results[i].tokens contains tokens sequence (with length T)
# results[i].score contains score of the hypothesis
# results is sorted array with the best hypothesis stored with index=0.
```

### Decoding with your own language model
One can define custom language model in python and use it for beam search decoding.

To store language model state, derive from the `LMState` base class and define additional data corresponding to each state by creating `dict(LMState, info)` inside the language model class:

```python
import numpy
from flashlight.lib.text.decoder import LM


class MyPyLM(LM):
    mapping_states = dict() # store simple additional int for each state

    def __init__(self):
        LM.__init__(self)

    def start(self, start_with_nothing):
        state = LMState()
        self.mapping_states[state] = 0
        return state

    def score(self, state : LMState, token_index : int):
        """
        Evaluate language model based on the current lm state and new word
        Parameters:
        -----------
        state: current lm state
        token_index: index of the word
                    (can be lexicon index then you should store inside LM the
                    mapping between indices of lexicon and lm, or lm index of a word)

        Returns:
        --------
        (LMState, float): pair of (new state, score for the current word)
        """
        outstate = state.child(token_index)
        if outstate not in self.mapping_states:
            self.mapping_states[outstate] = self.mapping_states[state] + 1
        return (outstate, -numpy.random.random())

    def finish(self, state: LMState):
        """
        Evaluate eos for language model based on the current lm state

        Returns:
        --------
        (LMState, float): pair of (new state, score for the current word)
        """
        outstate = state.child(-1)
        if outstate not in self.mapping_states:
            self.mapping_states[outstate] = self.mapping_states[state] + 1
        return (outstate, -1)
```

LMState is a C++ base class for language model state. Its `compare` method (for comparing one state with another) is used inside the beam search decoder.
It also has a `LMState child(int index)` method which returns a state obtained by following the token with this index from current state.

All LM states are organized as a trie. We use the `child` method in python to properly create this trie (which will be used inside the decoder to compare states) and can store additional state data in `mapping_states`.

This language model can be used as follows. Here, we print the state and its additional stored info inside `lm.mapping_states`:

```python
custom_lm = MyLM()

state = custom_lm.start(True)
print(state, custom_lm.mapping_states[state])

for i in range(5):
    state, score = custom_lm.score(state, i)
    print(state, custom_lm.mapping_states[state], score)

state, score = custom_lm.finish(state)
print(state, custom_lm.mapping_states[state], score)
```

and for the decoder:

```python
decoder = LexiconDecoder(options, trie, custom_lm, sil_idx, blank_inx, unk_idx, transitions, False)
```

## Tests and Examples

An integration test for Python decoder bindings can be found in `bindings/python/test/test_decoder.py`. To run, use:
```bash
cd bindings/python/test
python3 -m unittest discover -v .
```
