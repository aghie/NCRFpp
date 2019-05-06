# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-03-30 16:20:07

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from wordsequence import WordSequence
from crf import CRF
from collections import defaultdict
from torch.distributions import Categorical
from torch.autograd import Variable



class SeqModel_Policy(nn.Module):
    def __init__(self, data):
        super(SeqModel_Policy, self).__init__()
        self.use_crf = data.use_crf
        print "build network..."
        print "use_char: ", data.use_char
        if data.use_char:
            print "char feature extractor: ", data.char_feature_extractor
        print "word feature extractor: ", data.word_feature_extractor
        print "use crf: ", self.use_crf

        self.gpu = data.HP_gpu
        self.average_batch = data.average_batch_loss
        ## add two more label for downlayer lstm, use original label size for CRF

        label_size = {}
        for idtask in range(data.HP_tasks):
            label_size[idtask] = data.label_alphabet_sizes[idtask]
            data.label_alphabet_sizes[idtask]+= 2

   #     label_size = data.label_alphabet_size
   #     data.label_alphabet_sizes[idtask] += 2
        self.word_hidden = WordSequence(data)
        if self.use_crf:
            self.crf = {}
            for idtask in range(data.HP_tasks):
                self.crf = {idtask:CRF(label_size[idtask], self.gpu)}
            #self.crf = CRF(label_size, self.gpu)

        self.data = data
        self.tasks_weights = self.data.HP_tasks_weights


    def neg_log_likelihood_loss(self, word_inputs, feature_inputs, word_seq_lengths,
                                char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask, inference):
       #outs = self.word_hidden(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
       outs = self.word_hidden(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, inference)

       batch_size = word_inputs.size(0)
       seq_len = word_inputs.size(1)

       losses = []
       scores = []
       tag_seqs = []

       if self.use_crf:
      #     print self.crf
           for idtask,out in enumerate(outs):
               loss = self.crf[idtask].neg_log_likelihood_loss(out, mask, batch_label[idtask])
               score,tag_seq = self.crf[idtask]._viterbi_decode(out,mask)
               losses.append(self.tasks_weights[idtask]*loss)
               scores.append(score)
               tag_seqs.append(tag_seq)

       else:

           for idtask,out in enumerate(outs):


               loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
               out = out.view(batch_size * seq_len, -1)
               score = F.log_softmax(out,1)

               aux_loss = loss_function(score, batch_label[idtask].view(batch_size*seq_len))
               _, tag_seq  = torch.max(score, 1)
               tag_seq = tag_seq.view(batch_size, seq_len)

               losses.append(self.tasks_weights[idtask]*aux_loss)
               scores.append(score.sum())
               tag_seqs.append(tag_seq)

       total_loss = sum(losses)

       if self.average_batch:
           total_loss = total_loss / batch_size

       return total_loss, losses, tag_seqs, scores



    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask, inference):
        outs = self.word_hidden(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, inference)

        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        scores = []
        tag_seqs = []
        if self.use_crf:
            for idtask,out in enumerate(outs):
                score, tag_seq = self.crf[idtask]._viterbi_decode(out,mask)
                scores.append(score)
                tag_seqs.append(tag_seq)
        else:

            for idtask,out in enumerate(outs):
                out = out.view(batch_size * seq_len, -1)
                _, tag_seq = torch.max(out, 1)
                tag_seq = tag_seq.view(batch_size, seq_len)
                tag_seq = mask.long()*tag_seq
                tag_seqs.append(tag_seq)


        return tag_seqs


    def sample(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask, no_samples):
        all_outs = self.word_hidden(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, inference = False)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        all_samples_tag_sequences = [] #defaultdict(list)
        all_scores = [] #defaultdict(list)
        all_entropies = []
        for n in range(no_samples):
            samples_tag_sequences = []
            scores = []
            entropies = []
            for idtask,outs in enumerate(all_outs):
            #print("outs: ", idtask, outs.shape)
                if self.use_crf:
                    total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label[idtask])
                    non_log_score, tag_seq = self.crf._viterbi_decode(outs, mask)
                    #print(non_log_score.shape)
                    seq_score = 0
                    m = Categorical(non_log_score)
                    tag_seq = m.sample()
                    score = m.log_prob(tag_seq)
                    tag_seq = tag_seq.view(batch_size, seq_len)
                    samples_tag_sequences[idtask].append(tag_seq.squeeze())
                    seq_score = score.sum()
                    scores.append(seq_score)
                else:
                    outs = outs.view(batch_size * seq_len, -1)
                    #print("outs: re ", outs.shape)
                    #outs = torch.clamp(outs, max = 30)
                    #seq_score = 0
                    non_log_score = F.softmax(outs, 1)
                    #distribution
                    m = Categorical(non_log_score)
                    tag_seq = m.sample()
                    seq_score = m.log_prob(tag_seq)
                    tag_seq = tag_seq.view(batch_size, seq_len)
                    #print(idtask, "tag_seq_squee ",  tag_seq.shape)
                    samples_tag_sequences.append(tag_seq)
                    #compute entropy
                    entropy = (seq_score * torch.exp(seq_score)).sum()
                    entropies.append(entropy)
                    seq_score = seq_score.sum()
                    scores.append(seq_score)
            #outer loop, append list containing labels for all three tasks per sample
            all_samples_tag_sequences.append(samples_tag_sequences)
            all_scores.append(scores)
            all_entropies.append(entropies)
            #print("bef:", len(all_samples_tag_sequences))
        #add gold as sample
        #samples_tag_sequences = []
        #scores = []
        #try:
        #    for idtask in range(len(all_outs)):
        #        gold = batch_label[idtask]
                #print(gold, gold.squeeze())
        #        samples_tag_sequences.append(gold.view(batch_size, seq_len))
        #        score = m.log_prob(gold.squeeze())
        #        seq_score = score.sum()
        #        scores.append(seq_score)
        #    all_samples_tag_sequences.append(samples_tag_sequences)
        #    all_scores.append(scores)
        #except:
        #    pass
        #print("AFTER ", len(all_samples_tag_sequences))
            #n = score.size(0)
            #print(n)
            #ones = [2.0] * n
            #all_rewards[idtask].append(ones)
            #if self.average_batch:
            #total_loss = total_loss / batch_size
        return all_scores, all_samples_tag_sequences, all_entropies

    def sample_batch(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,batch_label, mask, no_samples):
       all_outs = self.word_hidden(word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths,char_seq_recover)
       print(all_outs)
       #numpy gymnastics
       all_outs = torch.stack(all_outs, dim = 3)
       #all_outs = np.moveaxis(all_outs, 1, 0).reshape(word_inputs.size(0), 3, word_inputs.size(1), -1)
       #print(all_outs)
       #all_outs = all_outs.tolist()
       #print("all outs ",all_outs[0].shape, all_outs[1].shape, all_outs[2].shape)
       #outs = torch.clamp(outs, max = 18)
       batch_size = word_inputs.size(0)
       seq_len = word_inputs.size(1)
       batch_all_samples_tag_sequences = []
       all_samples_tag_sequences = []
       batch_all_scores = []
       all_scores = []  #
       print(all_outs)
       for example in all_outs:
           print(example)
           for n in range(no_samples):
               samples_tag_sequences = []
               scores = []
               for idtask, outs in enumerate(example):
                   print("outs: ", idtask, outs.shape)
                   if self.use_crf:
                       total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label[idtask])
                       non_log_score, tag_seq = self.crf._viterbi_decode(outs, mask)
                       # print(non_log_score.shape)
                       seq_score = 0
                       m = Categorical(non_log_score)
                       tag_seq = m.sample()
                       score = m.log_prob(tag_seq)
                       tag_seq = tag_seq.view(batch_size, seq_len)
                       samples_tag_sequences[idtask].append(tag_seq.squeeze())
                       seq_score = score.sum()
                       scores.append(seq_score)
                   else:
                       outs = outs.view(batch_size * seq_len, -1)
                       #print("outs: re ", outs.shape)
                       #outs = torch.clamp(outs, max = 18)
                       #seq_score = 0
                       non_log_score = F.softmax(outs, 1)
                       #distribution
                       m = Categorical(non_log_score)
                       tag_seq = m.sample()
                       seq_score = m.log_prob(tag_seq)
                       tag_seq = tag_seq.view(batch_size, seq_len)
                       #print("tag_seq_squee ", tag_seq.shape)
                       samples_tag_sequences.append(tag_seq)
                       seq_score = seq_score.sum()
                       scores.append(seq_score)
                       #outer loop, append list containing labels for all three tasks per sample
               all_samples_tag_sequences.append(samples_tag_sequences)
               all_scores.append(scores)
            # add gold as sample
            #        samples_tag_sequences = []
            #       scores = []
            #      for idtask in range(len(all_outs)):
            #         samples_tag_sequences.append(batch_label[idtask])
            #        score = m.log_prob(batch_label[idtask].squeeze())
            #        seq_score = score.sum()
            #        scores.append(seq_score)
            #    all_samples_tag_sequences.append(samples_tag_sequences)
            #    all_scores.append(scores)

            # n = score.size(0)
            # print(n)
            # ones = [2.0] * n
            # all_rewards[idtask].append(ones)
           batch_all_samples_tag_sequences.append(all_samples_tag_sequences)
           batch_all_scores.append(all_scores)
           assert (len(batch_all_samples_tag_sequences) == batch_size)
        # if self.average_batch:
        # total_loss = total_loss / batch_size
       return batch_all_scores, batch_all_samples_tag_sequences


    #TODO: Update to multitasking
    def decode_nbest(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask, nbest):
        if not self.use_crf:
            print "Nbest output is currently supported only for CRF! Exit..."
            exit(0)
        outs = self.word_hidden(word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        scores, tag_seq = self.crf._viterbi_decode_nbest(outs, mask, nbest)
        return scores, tag_seq



