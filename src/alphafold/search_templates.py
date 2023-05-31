# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for building the input features for the AlphaFold model."""

import os
import sys
from typing import Mapping, Optional, Sequence
from absl import logging
from alphafold.data import parsers
from alphafold.data.tools import hhblits
from alphafold.data.tools import hhsearch
from alphafold.data.tools import jackhmmer
import numpy as np
import argparse
import pdb


parser = argparse.ArgumentParser(description = '''Get the host and pathogen species and their order for the matching.''')

parser.add_argument('--fasta_path', nargs=1, type= str, default=sys.stdin, help = 'Path to fasta.')
parser.add_argument('--output_dir', nargs=1, type= str, default=sys.stdin, help = 'Path to output dir.')
parser.add_argument('--jackhmmer_binary_path', nargs=1, type= str, default=sys.stdin, help = 'Path to jackhmmer.')
parser.add_argument('--hhsearch_binary_path', nargs=1, type= str, default=sys.stdin, help = 'Path to hhsearch.')
parser.add_argument('--uniref90_database_path', nargs=1, type= str, default=sys.stdin, help = 'uniref90_database_path.')
parser.add_argument('--pdb70_database_path', nargs=1, type= str, default=sys.stdin, help = 'pdb70_database_path.')



# Internal import (7716).
FeatureDict = Mapping[str, np.ndarray]

class TemplateSearch:
  """Searches for templates."""

  def __init__(self,
               jackhmmer_binary_path: str,
               hhsearch_binary_path: str,
               uniref90_database_path: str,
               pdb70_database_path: str):
    """Constructs a feature dict for a given FASTA file."""

    self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=uniref90_database_path)
    self.hhsearch_pdb70_runner = hhsearch.HHSearch(
        binary_path=hhsearch_binary_path,
        databases=[pdb70_database_path])
    self.uniref_max_hits = 10000

  def process(self, input_fasta_path: str, msa_output_dir: str) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    num_res = len(input_sequence)

    #First uniref90 is searchsed with jackhmmer to produce an MSA
    jackhmmer_uniref90_result = self.jackhmmer_uniref90_runner.query(
        input_fasta_path)[0]

    #This MSA is converted to a3m
    uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(
        jackhmmer_uniref90_result['sto'], max_sequences=self.uniref_max_hits)

    #And the a3m MSA is used to search pdb70
    hhsearch_result = self.hhsearch_pdb70_runner.query(uniref90_msa_as_a3m)

    #Save the hhsearch_result
    pdb70_out_path = os.path.join(msa_output_dir, 'pdb70_hits.hhr')
    with open(pdb70_out_path, 'w') as f:
      f.write(hhsearch_result)

    print('Written hhsearch_result for pdb70 to',msa_output_dir)


############MAIN##############
#Parse args
args = parser.parse_args()

template_search = TemplateSearch(
    jackhmmer_binary_path=args.jackhmmer_binary_path[0],
    hhsearch_binary_path=args.hhsearch_binary_path[0],
    uniref90_database_path=args.uniref90_database_path[0],
    pdb70_database_path=args.pdb70_database_path[0])

template_search.process(
    input_fasta_path=args.fasta_path[0],
    msa_output_dir=args.output_dir[0])


