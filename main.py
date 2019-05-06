# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-04-26 14:10:30

import time
import sys
import argparse
import random
import copy
import torch
import gc
import cPickle as pickle
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import tempfile
import os
from utils.metric import get_ner_fmeasure
from model.seqmodel import SeqModel
from model.seqmodel_policy_ac import SeqModel_Policy
from utils.data import Data
from tree2labels.utils import sequence_to_parenthesis
import math
from py2evalb import scorer
from py2evalb  import parser as evalbparser
from tree2labels.encoding2multitask import decode_int
#from tree2labels.encoding2multitask_int import decode_int
from tree2labels.evaluate import posprocess_labels
import codecs

from subprocess import PIPE,Popen
from StringIO import StringIO
#Uncomment/Comment these lines to determine when and which GPU(s) to use
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

seed_num = 17
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

reload(sys)
sys.setdefaultencoding('UTF8')


def data_initialization(data):
    data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir)
    data.build_alphabet(data.dev_dir)
    data.build_alphabet(data.test_dir)
    data.fix_alphabet()


def load_gold_trees(data):
    file = open(data.gold_train_trees, 'r')
    trees = file.readlines()
    return trees


def decoded_to_tree(input):

    all_sentences = []
    all_preds = []
    raw_sentences = input.split("\n\n")
    for raw_sentence in raw_sentences:
        lines = raw_sentence.split("\n")
        if len(lines) != 1:
            sentence = [tuple(l.split("\t")[0:2]) for l in lines]
            preds_sentence = posprocess_labels([l.split("\t")[2] for l in lines])
            all_sentences.append(sentence)
            all_preds.append(preds_sentence)
    parenthesized_trees = sequence_to_parenthesis(all_sentences,all_preds)#,None,None,None)
    return(parenthesized_trees[0])



def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover, inference=False):
    """
    input:
        pred_variable (batch_size, sent_len): pred tag result
        gold_variable (batch_size, sent_len): gold result variable
        mask_variable (batch_size, sent_len): mask variable
    """
    
    if inference:
        pred_variable = pred_variable[word_recover]
        mask_variable = mask_variable[word_recover]
        seq_len = pred_variable.size(1)
        mask = mask_variable.cpu().data.numpy()
        pred_tag = pred_variable.cpu().data.numpy()
        batch_size = mask.shape[0]
        pred_label = []

        for idx in range(batch_size):
            pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            pred_label.append(pred)
        return pred_label, None
    else:
        pred_variable = pred_variable[word_recover]
        gold_variable = gold_variable[word_recover]
        mask_variable = mask_variable[word_recover]
        batch_size = gold_variable.size(0)
        seq_len = gold_variable.size(1)
        mask = mask_variable.cpu().data.numpy()
        pred_tag = pred_variable.cpu().data.numpy()
        gold_tag = gold_variable.cpu().data.numpy()
        batch_size = mask.shape[0]
        pred_label = []
        gold_label = []
        for idx in range(batch_size):
            pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            assert(len(pred)==len(gold))
            pred_label.append(pred)
            gold_label.append(gold)
        return pred_label, gold_label


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    # print "word recover:", word_recover.size()
    # exit(0)
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    seq_len = pred_variable.size(1)
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    
    #Trying the exponential learning decay
    
#     lr = init_lr * np.exp(-0.2*epoch)  #init_lr * (0.1 ** (epoch // 20))
#     print " Learning rate is setted as:", lr
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
    
    #This is the regular decay
    lr = init_lr/(1+decay_rate*epoch)
    print " Learning rate is setted as:", lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer



#TODO: Need to deal with nbest
def evaluate(data, model, name, inference, nbest=None):
    """
    Evaluate MTL
    """
    
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == "test":
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print "Error: wrong evaluate name,", name
    ## set model in eval model
    model.eval()
    batch_size = 128#128 #For comparison against Vinyals et al. (2015)
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
     
    #D: Variable to collect the preds and gold prediction in multitask learning
    pred_labels = {idtask:[] for idtask in range(data.HP_tasks)}
    gold_labels = {idtask:[] for idtask in range(data.HP_tasks)}
     
    nbest_pred_labels = {idtask:[] for idtask in range(data.HP_tasks)}
    nbest_pred_scores = {idtask:[] for idtask in range(data.HP_tasks)}
    
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size 
        if end > train_num:
            end =  train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, \
        batch_label, mask  = batchify_with_label(instance, data.HP_gpu, inference, True)
        if nbest:
            scores, nbest_tag_seq = model.decode_nbest(batch_word,batch_features,
                                                       batch_wordlen, batch_char, 
                                                       batch_charlen, batch_charrecover, mask, 
                                                       inference, nbest)
            tag_seq = []
            
            for idtask, task_nbest_tag_seq in enumerate(nbest_tag_seq):   
                nbest_pred_result = recover_nbest_label(task_nbest_tag_seq, mask, data.label_alphabet[idtask], batch_wordrecover)
                nbest_pred_labels[idtask] += nbest_pred_result
                nbest_pred_scores[idtask] += scores[idtask][batch_wordrecover].cpu().data.numpy().tolist()
                tag_seq.append(task_nbest_tag_seq[:,:,0])
        else:
            tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask,
                            inference=inference)
        if not inference:
            for idtask, task_tag_seq in enumerate(tag_seq):
                pred_label, gold_label = recover_label(task_tag_seq, batch_label[idtask], mask, data.label_alphabet[idtask], 
                                                       batch_wordrecover, inference=inference)
                pred_labels[idtask]+=pred_label
                gold_labels[idtask]+=gold_label
        else:
            for idtask, task_tag_seq in enumerate(tag_seq):
                pred_label, _ = recover_label(task_tag_seq, None, mask, data.label_alphabet[idtask], 
                                                       batch_wordrecover, inference=inference)
                pred_labels[idtask]+=pred_label
  
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
     
    #Computing the score for each task
    tasks_results = []
    range_tasks = data.HP_tasks if not inference else data.HP_main_tasks
    for idtask in range(range_tasks):
        if not inference:
             acc, p, r, f = get_ner_fmeasure(gold_labels[idtask], pred_labels[idtask], data.tagScheme)
        else:
            acc, p, r, f = -1, -1, -1,-1
        
        if nbest:
            tasks_results.append((speed,acc,p,r,f,nbest_pred_labels[idtask], nbest_pred_scores[idtask]))
        else:
            tasks_results.append((speed,acc,p,r,f,pred_labels[idtask],nbest_pred_scores[idtask]))
    return tasks_results


def evaluate_tagseq(data, tag_seq, batch_label, mask, batch_wordrecover):
    """This for evaluating while training with policy gradient"""
    #D: Variable to collect the preds and gold prediction in multitask learning
    pred_labels = {idtask:[] for idtask in range(data.HP_tasks)}
    gold_labels = {idtask:[] for idtask in range(data.HP_tasks)}
    start_time = time.time()

    #TODO: It currently only works if all the tasks are the same
    if type(tag_seq) == type([]):
        for idtask, task_tag_seq in enumerate(tag_seq):
            pred_label, gold_label = recover_label(task_tag_seq, batch_label[idtask], mask, data.label_alphabet[idtask],
                                                   batch_wordrecover)
            pred_labels[idtask]+=pred_label
            gold_labels[idtask]+=gold_label
    else:
	for idtask, task_tag_seq in tag_seq.iteritems():
            pred_label, gold_label = recover_label(task_tag_seq[0], batch_label[idtask], mask, data.label_alphabet[idtask],
                                                   batch_wordrecover)
            pred_labels[idtask]+=pred_label
            gold_labels[idtask]+=gold_label

    decode_time = time.time() - start_time
    speed = len(tag_seq[0])/decode_time

    #D: Evaluating the different tasks
    tasks_results = []
    for idtask in range(0, data.HP_tasks):
        acc, p, r, f = get_ner_fmeasure(gold_labels[idtask], pred_labels[idtask], data.tagScheme)
        tasks_results.append((speed,acc,p,r,f,pred_labels[idtask]))
    return tasks_results


def batchify_with_label(input_batch_list, gpu, inference, volatile_flag=False):
    """
        input: list of words, chars and labels, various length. [[words,chars, labels],[words,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len) 
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order 
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len) 
    """
    
    batch_size = len(input_batch_list)
    
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0][0])
    chars = [sent[2] for sent in input_batch_list]
    if not inference:
        labels = [sent[3] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(map(len, words))
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long()
    #Creating n label_seq_tensors, one for each task
    
    if not inference:
        label_seq_tensor = {idtask: autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).long() 
                            for idtask in range(len(labels[0]))}
    else:
        label_seq_tensor = None
    
    
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).long())
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).byte()
    
    
    if not inference:
    
        for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
            word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            
            for idtask in label_seq_tensor:
                label_seq_tensor[idtask][idx, :seqlen] = torch.LongTensor(label[idtask])
           
            mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
            for idy in range(feature_num):
                feature_seq_tensors[idy][idx,:seqlen] = torch.LongTensor(features[idx][:,idy])
                
    else:

        for idx, (seq, seqlen) in enumerate(zip(words, word_seq_lengths)):
            word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
                       
            mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
            for idy in range(feature_num):
                feature_seq_tensors[idy][idx,:seqlen] = torch.LongTensor(features[idx][:,idy])
                        
                         
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

    if not inference:
        for idtask in label_seq_tensor:
            label_seq_tensor[idtask] = label_seq_tensor[idtask][word_perm_idx]
    
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [map(len, pad_char) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len)), volatile =  volatile_flag).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
    
    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        
        #label_seq_tensor = label_seq_tensor.cuda()
        if not inference:
            for idtask in label_seq_tensor:
                label_seq_tensor[idtask] = label_seq_tensor[idtask].cuda()
            
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    
    return word_seq_tensor,feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask



def train(data):
    print "Training model..."
    data.show_data_summary()
    save_data_name = data.model_dir +".dset"
    data.save(save_data_name)
    model = SeqModel(data)
    
    if data.pretrained_model is not None:
    
        model_dict = model.state_dict()
        
        #We load the weights for the layers that we have pretrained (e.g. for language modeling)
        pretrained_dict = torch.load(data.pretrained_model)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                           if data.pretrained_part == data.PRETRAINED_ALL or 
                           (data.pretrained_part == data.PRETRAINED_LSTMS and "hidden2tagList" not in k)}

        # We overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # We load the new state dict
        model.load_state_dict(model_dict)
    
    print "Model"
    print 
    print model.parameters
    print 

    loss_function = nn.NLLLoss()
    if data.optimizer.lower() == "sgd":
        #optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum)
        optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum,weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    else:
        print("Optimizer illegal: %s"%(data.optimizer))
        exit(0)
    best_dev = -10
    # data.HP_iteration = 1
    ## start training
   # optimizer = torch.optim.lr_scheduler.ExponentialLR(optimizer,data.HP_lr_decay)
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" %(idx,data.HP_iteration))
        if data.optimizer == "SGD":
            optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_id = 0
        sample_loss = 0
        total_loss = 0
        
        sample_loss = {idtask: 0 for idtask in range(data.HP_tasks)}
        right_token = {idtask: 0 for idtask in range(data.HP_tasks)}
        whole_token = {idtask: 0 for idtask in range(data.HP_tasks)}
        random.shuffle(data.train_Ids)
        ## set model in train model
        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num//batch_size+1
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size 
            if end >train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue
          
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, inference=False)
            instance_count += 1
#            loss, losses, tag_seq = model.neg_log_likelihood_loss(batch_word,batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
            loss, losses, tag_seq = model.neg_log_likelihood_loss(batch_word,batch_features, 
                                                                  batch_wordlen, batch_char, 
                                                                  batch_charlen, batch_charrecover, 
                                                                  batch_label, mask, inference=False)
            for idtask in range(data.HP_tasks):
                right, whole = predict_check(tag_seq[idtask], batch_label[idtask], mask)
                sample_loss[idtask]+= losses[idtask].data[0]
                right_token[idtask]+=right
                whole_token[idtask]+=whole
                if end%500 == 0:
                    temp_time = time.time()
                    temp_cost = temp_time - temp_start
                    temp_start = temp_time
                    print("     Instance: %s; Task %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, idtask, temp_cost, sample_loss[idtask], right_token[idtask], whole_token[idtask],(right_token[idtask]+0.)/whole_token[idtask]))
                    if sample_loss[idtask] > 1e8 or str(sample_loss) == "nan":
                        print "ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT...."
                        exit(0)
                    sys.stdout.flush()
                    sample_loss[idtask] = 0     
        
            if end%500 == 0:
                print "--------------------------------------------------------------------------"     

            total_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            model.zero_grad()
                                  
            
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        for idtask in range(data.HP_tasks):
            print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss[idtask], right_token[idtask], whole_token[idtask],(right_token[idtask]+0.)/whole_token[idtask]))       
         
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx, epoch_cost, train_num/epoch_cost, total_loss))
        print "totalloss:", total_loss
        if total_loss > 1e8 or str(total_loss) == "nan":
            print "ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT...."
            exit(0)
        # continue
         
        #Getting the summary results for all tasks
        summary = evaluate(data,model, "dev", False, False)     
        
 
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish
 
        current_scores = []
        for idtask in xrange(0, data.HP_tasks):
            speed,acc,p,r,f,pred_labels,_ = summary[idtask]
            if data.seg:
                current_score = f
                current_scores.append(f)
                print("Task %d Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(idtask, dev_cost, speed, acc, p, r, f))
            else:
                current_score = acc
                current_scores.append(acc)
                print("Task %d Dev: time: %.2fs speed: %.2fst/s; acc: %.4f"%(idtask, dev_cost, speed, acc))

        pred_results_tasks = []
        pred_scores_tasks = []
        for idtask in xrange(data.HP_tasks):
            speed, acc, p, r, f, pred_results, pred_scores = summary[idtask]
            pred_results_tasks.append(pred_results)
            pred_scores_tasks.append(pred_scores_tasks)

        
        if data.optimize_with_evalb:
        
            with tempfile.NamedTemporaryFile() as f_decode_mt:
                with tempfile.NamedTemporaryFile() as f_decode_st:
                    
                    #If we are learning multiple task we move it as a sequence labeling
                    if data.HP_main_tasks > 1: 
                        data.decode_dir = f_decode_mt.name
                        decoded_st_dir = f_decode_st.name
                        data.write_decoded_results(pred_results_tasks, 'dev')                   
                        os.system("PYTHONPATH="+data.tree2labels+" python "+data.en2mt+" --input "+data.decode_dir+" --output "+decoded_st_dir+" --status decode")     
                    
                    else:
                        
                        if data.decode_dir is None:
                            data.decode_dir = f_decode_st.name
                            decoded_st_dir =  f_decode_st.name
                        data.write_decoded_results(pred_results_tasks, 'dev')    
                        
                    command = ["PYTHONPATH="+data.tree2labels,"python",
                               data.evaluate," --input ",decoded_st_dir," --gold ",data.gold_dev_trees," --evalb ",data.evalb,">",f_decode_mt.name]
                  
                    os.system(" ".join(command))
                    current_score = float([l for l in f_decode_mt.read().split("\n")
                                       if l.startswith("Bracketing FMeasure")][0].split("=")[1])
                    print "Current Score (from EVALB)", current_score, "Previous best dev (from EVALB)", best_dev
       
                    
        else:
            print "Optimizing based on the accuracy"
            current_score = sum(current_scores) / len(current_scores)
            print "Current Score", current_score, "Previous best dev", best_dev
            
        if current_score > best_dev:
        #if current_score > best_dev:
            if data.seg:
                print "Exceed previous best f score:", best_dev
            else:
                print "Exceed previous best acc score:", best_dev
 
            model_name = data.model_dir +".model"
            print "Overwritting model to", model_name
            torch.save(model.state_dict(), model_name)
        
            best_dev = current_score 
        
        # ## decode test
        summary = evaluate(data,model, "test", False)        
        test_finish = time.time()
        test_cost = test_finish - dev_finish
 
        for idtask in xrange(0, data.HP_tasks):
            speed,acc,p,r,f,_,_ = summary[idtask]
            if data.seg:
                current_score = f
                print("Task %d Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(idtask, test_cost, speed, acc, p, r, f))
            else:
                current_score = acc
                print("Task %d Test: time: %.2fs speed: %.2fst/s; acc: %.4f"%(idtask, test_cost, speed, acc))
         
        gc.collect() 

        
def train_policy_grad(data):
    print("Training model with PG...")
    data.show_data_summary()
    save_data_name = data.model_dir +".dset"
    data.save(save_data_name)

    #load model for fine-tuning
    print("Load pre-trained model Model from file: ", data.pretrained_model)
    model = SeqModel_Policy(data)
    model.load_state_dict(torch.load(data.pretrained_model))
    print(model)
    #make baseline copy, freeze params
    model_base = copy.deepcopy(model)
    for param in model_base.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "word_hidden.lstm" in name or  "word_hidden.hidden2tagList.0" in name or "word_hidden.hidden2tagList.1" in name or  "word_hidden.hidden2tagList.3" in name or  "word_hidden.hidden2tagList.2" in name:
           param.requires_grad = True
           print(name)
        else:
           param.requires_grad = False

    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=data.HP_lr, momentum=data.HP_momentum,weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad,model.parameters()), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=data.HP_lr, weight_decay=data.HP_l2)
    else:
	print("Optimizer illegal: %s"%(data.optimizer))
        exit(1)

    best_dev = -10
    gold_train_trees = load_gold_trees(data)
    sum_advantage = 0
    sum_advantage_squared = 0
    count_advantage = 0
    total_count_train_instances = 0

    ## start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" %(idx,data.HP_iteration))

        instance_count = 0
        f1_running_base = 0
        f1_running_sample = 0
        f1_counter = 0
        f1_running_pol = 0

        #random.shuffle(data.train_Ids) #not shuffling because have to query training file in order

        model.train()
        model.zero_grad()

        batch_size = data.HP_batch_size
        no_samples = data.No_samples
        train_num = len(data.train_Ids)
        total_batch = train_num//batch_size+1

        for batch_id in range(total_batch):
             start = batch_id*batch_size
             end = (batch_id+1)*batch_size
             if end >train_num:
                 end = train_num
             instance = data.train_Ids[start:end]
             if not instance:
                 continue
             batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover,\
             batch_label, mask  = batchify_with_label(instance, data.HP_gpu,  inference=False)
             instance_count += 1
             total_count_train_instances += 1
             #baseline
             loss_base, losses_base, tag_seq_base, scores_base = \
                 model_base.neg_log_likelihood_loss(batch_word,batch_features, batch_wordlen, batch_char,
                                                    batch_charlen, batch_charrecover, batch_label, mask, inference=False)
             #sample
             try:
                 scores_samples, sample_seqs, all_seq_entropies = \
                     model.sample(batch_word,batch_features, batch_wordlen, batch_char, batch_charlen,
                                  batch_charrecover, batch_label, mask, no_samples)
             except:
                 sample_seqs = tag_seq_base
                 scores_samples =  scores_base

             #policy with max
             try:
                loss_pol, losses_pol, tag_seq_pol, _ = \
                    model.neg_log_likelihood_loss(batch_word,batch_features, batch_wordlen, batch_char,
                                                  batch_charlen, batch_charrecover, batch_label, mask, inference=False)
             except:
                tag_seq_pol = tag_seq_base
             #calc base f1
             summary_train_base =  evaluate_tagseq(data, tag_seq_base, batch_label, mask, batch_wordrecover)
             pred_results_tasks = []
             for idtask in xrange(data.HP_tasks):
                _, _, _, _, _, pred_results =  summary_train_base[idtask]
                pred_results_tasks.append(pred_results)
             #in python evalb
             #get pred tree
             preds_ids = data.write_decoded_results_ids_int(pred_results_tasks, train_id = batch_id)
             pred_str = decode_int(preds_ids)

             """
             pred_tree_str = decoded_to_tree(pred_str)
            # print "pred_tree_str", pred_tree_str
             gold_tree_str = gold_train_trees[batch_id]
            # print "gold_tree_str", gold_tree_str
             pred_tree = evalbparser.create_from_bracket_string(pred_tree_str)
            # print "pred_tree", pred_tree
             gold_tree = evalbparser.create_from_bracket_string(gold_tree_str)
            # print "gold_tree", gold_tree
             S = scorer.Scorer()
             result = str(S.score_trees(gold_tree, pred_tree))
               
             precision = float(result.split("prec:")[1].split()[0])
             recall = float(result.split("recall:")[1].split()[0])
             if precision == 0 and recall == 0:
                 current_score_base = 0
             else:
                 current_score_base = 2 * ((precision * recall) / (precision + recall))

             """
             try:
                 pred_tree_str = decoded_to_tree(pred_str)
            #     print "pred_tree_str", pred_tree_str
                 gold_tree_str = gold_train_trees[batch_id]
            #     print "gold_tree_str", gold_tree_str
                 pred_tree = evalbparser.create_from_bracket_string(pred_tree_str)
            #     print "pred_tree", pred_tree
                 gold_tree = evalbparser.create_from_bracket_string(gold_tree_str)
            #     print "gold_tree", gold_tree
                 S = scorer.Scorer()
                 result = str(S.score_trees(gold_tree, pred_tree))
               
                 precision = float(result.split("prec:")[1].split()[0])
                 recall = float(result.split("recall:")[1].split()[0])
                 if precision == 0 and recall == 0:
                     current_score_base = 0
                 else:
                     current_score_base = 2 * ((precision * recall) / (precision + recall))
             except:
                continue
            
             #calc pol f1
             summary_train_pol = evaluate_tagseq(data, tag_seq_pol, batch_label, mask, batch_wordrecover)
             pred_results_tasks = []
             for idtask in xrange(data.HP_tasks):
                 _, _, _, _, _, pred_results = summary_train_pol[idtask]
                 pred_results_tasks.append(pred_results)
             if True:
                 #in python evalb
                 #get pred tree
                 preds_ids = data.write_decoded_results_ids_int(pred_results_tasks, train_id = batch_id)
                 pred_str = decode_int(preds_ids)
                 try:
                    pred_tree_str = decoded_to_tree(pred_str)
                    gold_tree_str = gold_train_trees[batch_id]
                    pred_tree = evalbparser.create_from_bracket_string(pred_tree_str)
                    gold_tree = evalbparser.create_from_bracket_string(gold_tree_str)
                    S = scorer.Scorer()
                    result = str(S.score_trees(gold_tree, pred_tree))
                    precision = float(result.split("prec:")[1].split()[0])
                    recall = float(result.split("recall:")[1].split()[0])  
                 except:
                    continue
             #start using the samples
             Js = Variable(torch.FloatTensor(np.zeros((len(sample_seqs)), dtype='float64')))
             if data.HP_gpu:
                 Js = Js.cuda()
             for index, (sample_seq, seq_score, seq_entropies) in enumerate(zip(sample_seqs, scores_samples, all_seq_entropies)):
                 # calc sample f1
                 summary_sample = evaluate_tagseq(data, sample_seq, batch_label, mask, batch_wordrecover)
                 pred_results_tasks = []
                 seq_total_entropy = 0
                 for idtask in xrange(data.HP_tasks):
                     _, _, _, _, _, pred_results = summary_sample[idtask]
                     pred_results_tasks.append(pred_results)
                     f1_counter += 1
                     seq_total_entropy += seq_entropies[idtask]
                 if True:
                    # in python evalb
                    # get pred tree
                     preds_ids = data.write_decoded_results_ids_int(pred_results_tasks, train_id=batch_id)
                     pred_str = decode_int(preds_ids)
                     try:
                         pred_tree_str = decoded_to_tree(pred_str)
                         gold_tree_str = gold_train_trees[batch_id]
                         pred_tree = evalbparser.create_from_bracket_string(pred_tree_str)
                         gold_tree = evalbparser.create_from_bracket_string(gold_tree_str)
                         S = scorer.Scorer()
                         result = str(S.score_trees(gold_tree, pred_tree))
                         precision = float(result.split()[7].split(':')[1])
                         recall = float(result.split()[6].split(':')[1])
                         current_score_sample = 2 * ((precision * recall) / (precision + recall))
                     except:
                         continue

                 # rewards and advantage
                 reward_sample = Variable(torch.FloatTensor(np.array([float(current_score_sample)], dtype='float64')),
                                         requires_grad=False)
                 reward_base = Variable(torch.FloatTensor(np.array([float(current_score_base)], dtype='float64')),
                                       requires_grad=False)
                 if data.HP_gpu:
                     reward_sample = reward_sample.cuda()
                     reward_base = reward_base.cuda()

                 advantage = reward_sample  - (reward_base)
                 # moving avg
                 sum_advantage += advantage
                 sum_advantage_squared += (advantage * advantage)
                 count_advantage += 1
                 if data.pg_variance_reduce and count_advantage > data.variance_reduce_burn_in:
                     mean = sum_advantage / count_advantage
                     stdev = math.sqrt((sum_advantage_squared / count_advantage) - (mean * mean))
                     advantage = (advantage - mean) /  stdev
                 if data.entropy_regularisation:
                     J = - sum(seq_score) * advantage + seq_total_entropy * float(data.entropy_reg_coeff) 
                 else:
                     J = - sum(seq_score) * advantage 
                 if data.HP_gpu:
                     J = J.cuda()
                 Js[index] = J
             #backprop
             Js.backward(torch.ones_like(Js))
             grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), 5.)
             optimizer.step()
             model.zero_grad()

             if end%int(data.pg_valsteps) == 0:
                 temp_time = time.time()
                 temp_cost = temp_time - temp_start
                 temp_start = temp_time
                 print("Instance: %s; Time: %.2fs; f1_base: %.4f; f1_sample: %.4f;  f1_pol: %.4f" %
                       (end, temp_cost, f1_running_base, f1_running_sample, f1_running_pol))
                 #reset
                 f1_running_sample = f1_running_base = f1_running_pol = 0
                 f1_counter = 0

             #eval on dev for model selection
             if end%int(data.pg_valsteps) == 0:
                 print("--------------------------------------------------------------------------")
                 summary = evaluate(data, model, "dev", False)
                 pred_results_tasks = []
                 pred_scores_tasks = []
                 for idtask in xrange(data.HP_tasks):
                     speed, acc, p, r, f, pred_results, pred_scores = summary[idtask]
                     pred_results_tasks.append(pred_results)
                     pred_scores_tasks.append(pred_scores_tasks)

                 if True:  # data.optimize_evalb:
                     with tempfile.NamedTemporaryFile() as f_decode_mt:
                         with tempfile.NamedTemporaryFile() as f_decode_st:
                             data.decode_dir = f_decode_mt.name
                             decoded_st_dir = f_decode_st.name
                             data.write_decoded_results(pred_results_tasks, 'dev')
                             os.system(
                                 "python " + data.en2mt + " --input " + data.decode_dir + " --output " +
                                 decoded_st_dir + " --status decode")
                             
                             command = ["PYTHONPATH=" + data.tree2labels, "python",
                                        data.evaluate, " --input ", decoded_st_dir, " --gold ", data.gold_dev_trees,
                                        " --evalb ", data.evalb, ">", f_decode_mt.name]
                             os.system(" ".join(command))
                             
                             
                             current_score = float(
                                 [l for l in f_decode_mt.read().split("\n") if l.startswith("Bracketing FMeasure")][0].split("=")[1])
                             print("Current Score (from EVALB)", current_score, "Previous best dev (from EVALB)",
                                   best_dev)

                 if current_score > best_dev:
                     if data.seg:
                         print("Exceed previous best f score:", best_dev)
                     else:
                         print("Exceed previous best acc score:", best_dev)
                     model_name = data.model_dir + '.' + str(idx) + ".model"
                     print("Save current best model in file:", model_name)
                     torch.save(model.state_dict(), model_name)
                     best_dev = current_score

        #end of epoch
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s"%(idx, epoch_cost, train_num/epoch_cost))
        gc.collect()



def load_model_decode(data, name):
    print "Load Model from file: ", data.model_dir
    #model = SeqModel(data)
    model = SeqModel_Policy(data)
    model.load_state_dict(torch.load(data.load_model_dir))

    print("Decode %s data, nbest: %s ..."%(name, data.nbest))
    start_time = time.time()
    
    summary = evaluate(data, model, name, True, data.nbest)
    pred_results_tasks = []
    pred_scores_tasks = []
    range_tasks = data.HP_main_tasks
    #for idtask in xrange(data.HP_tasks):
    for idtask in xrange(range_tasks):
        speed, acc, p, r, f, pred_results, pred_scores = summary[idtask]
        pred_results_tasks.append(pred_results)
        pred_scores_tasks.append(pred_scores)
    end_time = time.time()
    time_cost = end_time - start_time
    if data:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(name, time_cost, speed, acc, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f"%(name, time_cost, speed, acc))
        
    return pred_results_tasks, pred_scores_tasks                    

def load_models_avg_decode(data, name):
    """computes average of three models' parameters and decodes based on that"""
    print "Load Models from file: ", data.model_dir
    model =  SeqModel_Policy(data)
    model1 = copy.deepcopy(model)
    model2 =  copy.deepcopy(model)
    model3 = copy.deepcopy(model)

    model1.load_state_dict(torch.load(data.load_model_dir1))
    model2.load_state_dict(torch.load(data.load_model_dir2))
    model3.load_state_dict(torch.load(data.load_model_dir3))

    #interpolate
    beta = 0.333 #The interpolation parameter    
    params1 = model1.named_parameters()
    params2 = model2.named_parameters()
    params3 = model3.named_parameters()

    dict_params3 = dict(params3)
    for (name1, param1), (name2, param2) in zip(params1, params2):
        if name1 in dict_params3:
            dict_params3[name1].data.copy_(beta*param1.data + beta*param2.data + beta*dict_params3[name1].data)

    model.load_state_dict(dict_params3)

    print("Decode %s data, nbest: %s ..."%(name, data.nbest))
    start_time = time.time()

    summary = evaluate(data, model, name, True, data.nbest)
    pred_results_tasks = []
    pred_scores_tasks = []
    range_tasks = data.HP_main_tasks
    #for idtask in xrange(data.HP_tasks):
    for idtask in xrange(range_tasks):
        speed, acc, p, r, f, pred_results, pred_scores = summary[idtask]
        pred_results_tasks.append(pred_results)
        pred_scores_tasks.append(pred_scores)
    end_time = time.time()
    time_cost = end_time - start_time
    if data:
	print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(name, time_cost, speed, acc, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f"%(name, time_cost, speed, acc))

    return pred_results_tasks, pred_scores_tasks




# def load_model_decode(data, name):
#     print "Load Model from file: ", data.model_dir
#     model = SeqModel(data)
#     model.load_state_dict(torch.load(data.load_model_dir))
# 
#     print("Decode %s data, nbest: %s ..."%(name, data.nbest))
#     start_time = time.time()
#     speed, acc, p, r, f, pred_results, pred_scores = evaluate(data, model, name, data.nbest)
# 
#     end_time = time.time()
#     time_cost = end_time - start_time
#     if data:
#         print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(name, time_cost, speed, acc, p, r, f))
#     else:
#         print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f"%(name, time_cost, speed, acc))
#     return pred_results, pred_scores




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with NCRF++')
    # parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--config',  help='Configuration File' )
    
    args = parser.parse_args()
    data = Data()
    data.read_config(args.config)
    status = data.status.lower()
    data.HP_gpu = torch.cuda.is_available()
    print "Seed num:",seed_num
    
    if status == 'train':
        print("MODEL: train")
        data_initialization(data)
        data.generate_instance('train')
        data.generate_instance('dev')
        data.generate_instance('test')
        data.build_pretrain_emb()
        train(data)
    elif status == 'finetune':
        print("MODEL: finetune")
        data_initialization(data)
        data.generate_instance('train')
        data.generate_instance('dev')
        data.generate_instance('test')
        data.build_pretrain_emb()
        train_policy_grad(data)
    elif status == 'decode':   
        print("MODEL: decode")
        data.load(data.dset_dir)  
        data.read_config(args.config) 
        print data.raw_dir
        # exit(0) 
        data.show_data_summary()
        data.generate_instance('raw')
        print("nbest: %s"%(data.nbest))
        decode_results, pred_scores = load_model_decode(data, 'raw')
        #decode_results, pred_scores = load_models_avg_decode(data, 'raw')  
        if data.nbest:
            data.write_nbest_decoded_results(decode_results, pred_scores, 'raw')
        else:
            data.write_decoded_results(decode_results, 'raw')
    else:
        print "Invalid argument! Please use valid arguments! (train/test/finetune/decode)"




