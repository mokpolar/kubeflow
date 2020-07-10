import tensorflow.keras.models as M
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow import keras
import tensorflow as tf


class SequenceModel(M.Model):
    def __init__(self, n_class=6):
        super(SequenceModel, self).__init__()
        
        self.n_class = n_class
        self.n_hidden = 96
        dropout_rate = 0.5
        
        ###### Seq Model1 #######
        self.fea_conv = M.Sequential([L.Dropout(dropout_rate),
                                      L.Conv2D(512, 1,padding='same', use_bias=False),
                                      L.BatchNormalization(),
                                      L.ReLU(),
                                      L.Dropout(dropout_rate),
                                      L.Conv2D(128, 1, padding='same', use_bias=False),
                                      L.BatchNormalization(),
                                      L.ReLU(),
                                      L.Dropout(dropout_rate)])
        
        self.fea_first_final = M.Sequential([L.Conv2D(self.n_class, 1, padding='same')])
        
        
        ###### Bidirectional GRU1 ####### num_layers in pytorch?
        self.fea_gru = L.Bidirectional(L.GRU(self.n_hidden, return_sequences=True))
        self.fea_gru_final = M.Sequential([L.Conv2D(self.n_class, kernel_size=(1, self.n_hidden*2))])

        ###### Seq Model2 ########
        self.conv_first = M.Sequential([L.Conv2D(128, kernel_size=(5,1), padding='same', use_bias=False),
                                        L.BatchNormalization(),
                                        L.ReLU(),
                                        L.Conv2D(64, kernel_size=(3,1), padding='same', use_bias=False, dilation_rate=2),
                                        L.BatchNormalization(),
                                        L.ReLU()])
        
        self.conv_res = M.Sequential([L.Conv2D(64, kernel_size=(3,1), padding='same', use_bias=False, dilation_rate=4),
                                      L.BatchNormalization(),
                                      L.ReLU(),
                                      L.Conv2D(64, kernel_size=(3,1), padding='same', use_bias=False, dilation_rate=2),
                                      L.BatchNormalization(),
                                      L.ReLU()])

        self.conv_final = M.Sequential([L.Conv2D(self.n_class, kernel_size=(3,1), padding='same', use_bias=False)])
        
        ####### Biderectional GRU2 ########
        self.gru = L.Bidirectional(L.GRU(self.n_hidden, return_sequences=True))
        self.final = M.Sequential(L.Conv2D(self.n_class, kernel_size=(1, self.n_hidden*2)))
        
    def call(self, x):
        features = x[0]
        x = x[1]
        batch_size, _, _, _ = features.shape
        
        ############### Seq1 ################
        # stem_fc
        x_fc = self.fea_conv(features) # (N, LenSeq, 1, Lenfeat)

        # fc
        out11 = self.fea_first_final(x_fc) # (N, LenSeq, 1, 6)

        # gru after reshape
        x_fc = K.squeeze(x_fc, 2) # (N, LenSeq, LenFeat)
        
        x_gru = self.fea_gru(x_fc) # (N, LenSeq, 192)
        x_gru = K.expand_dims(x_gru, axis=-1) # (N, LenSeq, 192, 1)
        
        # fc after lstm
        out12 = self.fea_gru_final(x_gru) # (N, LenSeq, 1, 6)
        
        # seq1 output (Elementwise Sum)
        out1 = out11 + out12
        out1_sigmoid = K.sigmoid(out1) # (N, LenSeq, 1, 6)

        # concat cnn out, seq1 out
        x = K.concatenate([x, out1_sigmoid], axis=-1) # (N, LenSeq, 1, 12)
        
        ############### Seq2 ################
        # stem_fc
        x = self.conv_first(x) # (N, LenSeq, 1, 64)
        x = self.conv_res(x) # (N, LenSeq, 1, 64)
               
        # fc
        out21 = self.conv_final(x) # (N, LenSeq, 1, 6)
        
        # gru after reshape
        x = K.squeeze(x, 2)# (N, LenSeq, 64)
        
        x = self.gru(x)  # (N, LenSeq, 192)
        x = K.expand_dims(x, axis=-1) # (N, LenSeq, 192, 1)
        
        # fc after lstm
        out22 = self.final(x) #(N, LenSeq, 1, 6)
        
        # seq2 output (Elementwise Sum)
        out2 = out21 + out22 # (N, LenSeq, 1, 6)

        return out1, out2
    
    def summary(self):
        x_features = L.Input((30, 1, 1024)) # [N, SeqLen, 1, LenFeature]
        x_inputs = L.Input((30, 1, self.n_class)) # [N, LenSeq, 1, LenOutp]
        M.Model([x_features, x_inputs], self.call((x_features, x_inputs))).summary()
        
        
if __name__=='__main__':
    
    import numpy as np
    
    model = SequenceModel(n_class=6)
    model.summary()
    model.save("./rnn_save")

    
    x_fea = np.ones((1, 40, 1, 1024)) # [N, SeqLen, 1, LenFeature]
    x_inp = np.ones((1, 40, 1, 6)) # [N, LenSeq, 1, LenOutp]

    out1, out2 = model.predict((x_fea, x_inp))
    print(out1.shape, out2.shape) # [N, LenSeq, 1, LenOutp]
