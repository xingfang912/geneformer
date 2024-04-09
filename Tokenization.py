# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:15:27 2024

@author: xfang
"""

class Tokenizer:
    def __init__(self):
        self.merges = None
        self.encodings = None
        
    def _get_stats(self, ids):
        count = {}
        for pair in zip(ids,ids[1:]):
            count[pair] = count.get(pair,0)+1
        return count    
    
    def _merge(self, ids, pair, idx):
        # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def train(self, train_data, final_vocab_size=276):
        """
        The train method applies BPE onto the train_data
        to obtain a merge table.
        """
        assert type(train_data) == str, "training data must be a string"
        assert final_vocab_size > 256, "final vocabulary size must be greater than 256"
        
        num_merges = final_vocab_size - 256
        
        #step 1: get the tokens for merging
        tokens = list(train_data.encode("utf-8"))

        from timeit import default_timer as timer 
        
        
        #step 2: merge token pairs based on their frequency
        
        merges = {} # (int, int) -> int
        for i in range(num_merges):
            
            start = timer()

            stats = self._get_stats(tokens)
            
            # selct the most frequent pair to merge
            pair = max(stats, key=stats.get)
            
            idx = 256 + i
            print(f"merging {pair} into a new token {idx}",end=" ,")
            tokens = self._merge(tokens, pair, idx)
            merges[pair] = idx

            end = timer()
            print(f"merging takes {end-start} seconds.")

            
        self.merges = merges
        
        
        
        
    def encode(self, input_data):
        assert self.merges != None, "merge table cannot be None for encoding"
        
        tokens = list(input_data.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self._get_stats(tokens)
            # find the most eligible pair to merge
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged
            idx = self.merges[pair]
            tokens = self._merge(tokens, pair, idx)
        
        self.encodings = tokens    
        return tokens
    
    def decode(self):
        assert self.encodings != None, "encodings cannot be None for decoding"
        
        vocab = {idx: bytes([idx]) for idx in range(256)} # a look-up table from integer to raw bytes
        # the vacob works fine for ascii characters, but we need to take care of other chars:
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]     

        # given ids (list of integers), return Python string
        tokens = b"".join(vocab[idx] for idx in self.encodings)
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    
class GeneTokenizer:
    def __init__(self):
        self.merges = None
        self.encodings = None
        self.geneformer_vocab_size = 25426
        self.current_vocab_size = 0
        self.token_list = None
        self.merges = {} # (int, int) -> int
        
    def _get_stats(self, id_list):
        count = {}
        for ids in id_list:
            for pair in zip(ids,ids[1:]):
                count[pair] = count.get(pair,0)+1
        return count    
    
    def _merge(self, token_list, pair, idx):
        new_token_list = []
        for ids in token_list:
            newids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                    newids.append(idx)
                    i += 2
                else:
                    newids.append(ids[i])
                    i += 1
            new_token_list.append(newids)        
        return new_token_list
    
    def train(self, train_data, final_vocab_size, save_steps=20):
        """
        train_data is a list of gene lists: [[2,3,7,...],...,[1813,975,...]]
        """
        if self.current_vocab_size == 0:
            self.current_vocab_size = self.geneformer_vocab_size
        
        
        num_merges = final_vocab_size - self.current_vocab_size
        
        assert num_merges > 0, f"number of merges must be positive, but found number of merges = {num_merges}"

        if num_merges < save_steps:
            print("Warning: The tokenizer may not be saved due to less number of merges than requried save steps!")
        
        #step 1: get the tokens for merging
        if self.token_list is None:
            self.token_list = train_data[:] # shallow copy

        else:
            pass # this is for the contiuation of training    
          
        
        #step 2: merge token pairs based on their frequency

        from timeit import default_timer as timer
        
        print(f"Run {num_merges} merges:")
        for i in range(num_merges):

            start = timer()

            stats = self._get_stats(self.token_list)
            
            # select the most frequent pair to merge
            pair = max(stats, key=stats.get)
            
            idx = self.current_vocab_size
            
            # display a message about what is being merged 
            print(f"merging {pair} into a new token {idx}",end=", ")

            self.token_list = self._merge(self.token_list, pair, idx)

            end = timer()

            print(f"merging takes {end-start:.2f} seconds.")

            self.merges[pair] = idx

            # increase the vocab size
            self.current_vocab_size += 1

            # save the tokenizer per 20 merges:
            if (i+1) % save_steps == 0:
                import pickle
                print("Saving the tokenizer...")
                pickle.dump(self,open("./saved_tokenizers/gene_tokenizer_"+str(self.current_vocab_size),"wb"))
                print("Saving complete!")

        # Final saving
        import pickle
        print("Saving the final tokenizer...")
        pickle.dump(self,open("./saved_tokenizers/gene_tokenizer_"+str(self.current_vocab_size),"wb"))
        print("Saving complete!")        
        
        
    def encode(self, input_data):
        assert self.merges != {}, "merge table cannot be empty for encoding"
        
        token_list = input_data[:] # a list of lists

        new_token_list = []
        for tokens in token_list:
            while len(tokens) >= 2:
                stats = self._get_stats([tokens])
                # find the most eligible pair to merge
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break # nothing else can be merged
                idx = self.merges[pair]
                tokens = self._merge([tokens], pair, idx)[0]

            new_token_list.append(tokens)
        
        self.encodings = new_token_list
        return new_token_list
    
    def decode(self):
        assert self.encodings != None, "encodings cannot be None for decoding"
        
        vocab = {idx: [idx] for idx in range(25426)}

        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]     

        # # given ids (list of integers), return Python string
        # tokens = b"".join(vocab[idx] for idx in self.encodings)
        # text = tokens.decode("utf-8", errors="replace")

        decodings = [[real_id for idx in tokens for real_id in vocab[idx]] for tokens in self.encodings]
        return decodings