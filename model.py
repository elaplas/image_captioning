import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_inputs = embed_size
        self.n_hidden = hidden_size
        self.n_outputs = vocab_size
        self.n_layers = num_layers
        self.lstm = nn.LSTM(self.n_inputs, self.n_hidden, self.n_layers, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(self.n_hidden, self.n_outputs)
        self.embedd=nn.Embedding(vocab_size, embed_size)
        self.dropout=nn.Dropout(p=0.5)
        self.norm=nn.BatchNorm2d(1)
        self.hidden=None
        #self.init_weights()
        
    
    def forward(self, features, captions):
        #concatenate inputs
        captions = captions[:, :-1]
        captions = self.embedd(captions.long())
        features = features.unsqueeze(1)
        x=torch.cat((features, captions), 1)
        # pass forward
        x = self.predict(x)
        return x
    
    def predict(self,x):
        
        if self.hidden==None:
            self.hidden=self.init_hidden(x.shape[0])
            
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        self.hidden = tuple([each.data for each in self.hidden]) 
        x, self.hidden = self.lstm(x, self.hidden)
        x = self.dropout(x)
        n_batch=x.shape[0]
        n_seq=x.shape[1]
        x = self.fc(x.view(n_batch*n_seq, -1))
        x=x.view(n_batch, n_seq, -1)
        return x
        
              
    def init_hidden(self, n_seqs=1):
        weight=next(self.parameters()).data
        return (weight.new(self.n_layers, n_seqs, self.n_hidden).zero_(), 
                weight.new(self.n_layers, n_seqs, self.n_hidden).zero_())
    
    def init_weights(self):
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-1, 1)
        self.embedd.weight.data.uniform_(-1,1)
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids          of length max_len) "
        predicted_caption=[]
        if states != None:
            self.hidden = states
        
        for i in range(max_len):
            p_words = self.predict(inputs)
            #p_words = p_words.cpu()
            #p_top_words, top_words = p_words.topk(1)
            #p_top_words = p_top_words.detach().numpy().squeeze()
            #top_words = top_words.detach().numpy().squeeze() 
            #predicted_word= np.random.choice(top_words, p = p_top_words/p_top_words.sum())
            predicted_word =  p_words.max(-1)[1]
            predicted_caption.append(int(predicted_word))
            if (int(predicted_word) ==1):
                break         
            inputs = torch.tensor([int(predicted_word)]).unsqueeze(0)
            inputs = inputs.cuda()
            inputs = self.embedd(inputs)     
        return predicted_caption
        
                  
            