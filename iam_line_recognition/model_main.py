import torchvision
import torch.nn as nn
import torch.nn.functional as F

from model_visual_features import ResNetFeatureExtractor, TPS_SpatialTransformerNetwork

class HW_RNN_Seq2Seq(nn.Module):
    """
    Visual Seq2Seq model using BiLSTM
    """
    def __init__(self, num_classes, image_height, cnn_output_channels=512, num_feats_mapped_seq_hidden=128, num_feats_seq_hidden=256):
        """
        ---------
        Arguments
        ---------
        num_classes : int
            num of distinct characters (classes) in the dataset
        image_height : int
            image height
        cnn_output_channels : int
            number of channels output from the CNN visual feature extractor (default: 512)
        num_feats_mapped_seq_hidden : int
            number of features to be used in the mapped visual features as sequences (default: 128)
        num_feats_seq_hidden : int
            number of features to be used in the LSTM for sequence modeling (default: 256)
        """
        super().__init__()
        self.output_height = image_height // 32

        self.dropout = nn.Dropout(p=0.25)
        self.map_visual_to_seq = nn.Linear(cnn_output_channels * self.output_height, num_feats_mapped_seq_hidden)

        self.b_lstm_1 = nn.LSTM(num_feats_mapped_seq_hidden, num_feats_seq_hidden, bidirectional=True)
        self.b_lstm_2 = nn.LSTM(2 * num_feats_seq_hidden, num_feats_seq_hidden, bidirectional=True)

        self.final_dense = nn.Linear(2 * num_feats_seq_hidden, num_classes)

    def forward(self, visual_feats):
        visual_feats = visual_feats.permute(3, 0, 1, 2)
        # WBCH
        # the sequence is along the width of the image as a sentence

        visual_feats = visual_feats.contiguous().view(visual_feats.shape[0], visual_feats.shape[1], -1)
        # WBC

        seq = self.map_visual_to_seq(visual_feats)
        seq = self.dropout(seq)
        lstm_1, _ = self.b_lstm_1(seq)
        lstm_2, _ = self.b_lstm_2(lstm_1)
        lstm_2 = self.dropout(lstm_2)

        dense_output = self.final_dense(lstm_2)
        # [seq_len, B, num_classes]

        log_probs = F.log_softmax(dense_output, dim=2)

        return log_probs


class CRNN(nn.Module):
    """
    Hybrid CNN - RNN model
    CNN - Modified ResNet34 for visual features
    RNN - BiLSTM for seq2seq modeling
    """
    def __init__(self, num_classes, image_height, num_feats_mapped_seq_hidden=128, num_feats_seq_hidden=256):
        """
        ---------
        Arguments
        ---------
        num_classes : int
            num of distinct characters (classes) in the dataset
        image_height : int
            image height
        num_feats_mapped_seq_hidden : int
            number of features to be used in the mapped visual features as sequences (default: 128)
        num_feats_seq_hidden : int
            number of features to be used in the LSTM for sequence modeling (default: 256)
        """
        super().__init__()
        self.visual_feature_extractor = ResNetFeatureExtractor()
        self.rnn_seq2seq_module = HW_RNN_Seq2Seq(num_classes, image_height, self.visual_feature_extractor.output_channels, num_feats_mapped_seq_hidden, num_feats_seq_hidden)

    def forward(self, x):
        visual_feats = self.visual_feature_extractor(x)
        # [B, 512, H/32, W/32]

        log_probs = self.rnn_seq2seq_module(visual_feats)
        return log_probs


class STN_CRNN(nn.Module):
    """
    STN + CNN + RNN model
    STN - Spatial Transformer Network for learning variable handwriting
    CNN - Modified ResNet34 for visual features
    RNN - BiLSTM for seq2seq modeling
    """
    def __init__(self, num_classes, image_height, image_width, num_feats_mapped_seq_hidden=128, num_feats_seq_hidden=256):
        """
        ---------
        Arguments
        ---------
        num_classes : int
            num of distinct characters (classes) in the dataset
        image_height : int
            image height
        image_width : int
            image width
        num_feats_mapped_seq_hidden : int
            number of features to be used in the mapped visual features as sequences (default: 128)
        num_feats_seq_hidden : int
            number of features to be used in the LSTM for sequence modeling (default: 256)
        """
        super().__init__()
        self.stn = TPS_SpatialTransformerNetwork(
            80,
            (image_height, image_width),
            (image_height, image_width),
            I_channel_num=3,
        )
        self.visual_feature_extractor = ResNetFeatureExtractor()
        self.rnn_seq2seq_module = HW_RNN_Seq2Seq(num_classes, image_height, self.visual_feature_extractor.output_channels, num_feats_mapped_seq_hidden, num_feats_seq_hidden)

    def forward(self, x):
        stn_output = self.stn(x)
        visual_feats = self.visual_feature_extractor(stn_output)
        log_probs = self.rnn_seq2seq_module(visual_feats)
        return log_probs

"""
class STN_PP_CRNN(nn.Module):
    def __init__(self, num_classes, image_height, image_width, num_feats_mapped_seq_hidden=128, num_feats_seq_hidden=256):
        super().__init__()
        self.stn = TPS_SpatialTransformerNetwork(
            20,
            (image_height, image_width),
            (image_height, image_width),
            I_channel_num=3,
        )
        self.visual_feature_extractor = ResNetFeatureExtractor()
        self.pp_block = PyramidPoolBlock(num_channels=self.visual_feature_extractor.output_channels)
        self.rnn_seq2seq_module = HW_RNN_Seq2Seq(num_classes, image_height, self.visual_feature_extractor.output_channels, num_feats_mapped_seq_hidden, num_feats_seq_hidden)

    def forward(self, x):
        stn_output = self.stn(x)
        visual_feats = self.visual_feature_extractor(stn_output)
        pp_feats = self.pp_block(visual_feats)
        log_probs = self.rnn_seq2seq_module(pp_feats)
        return log_probs
"""
