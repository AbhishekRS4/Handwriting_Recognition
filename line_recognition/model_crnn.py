import torchvision
import torch.nn as nn

from model_visual_features import ResNetFeatureExtractor

class CRNN(nn.Module):
    def __init__(self, num_classes, image_height, num_feats_mapped_seq_hidden=64, num_feats_seq_hidden=256):
        super().__init__()
        self.visual_feature_extractor = ResNetFeatureExtractor()
        self.output_height = image_height // 32

        self.map_visual_to_seq = nn.Linear(self.visual_feature_extractor.output_channels * self.output_height, num_feats_mapped_seq_hidden)

        self.b_lstm_1 = nn.LSTM(num_feats_mapped_seq_hidden, num_feats_seq_hidden, bidirectional=True)
        self.b_lstm_2 = nn.LSTM(2 * num_feats_seq_hidden, num_feats_seq_hidden, bidirectional=True)

        self.final_dense = nn.Linear(2 * num_feats_seq_hidden, num_classes)

    def forward(self, x):
        visual_feats = self.visual_feature_extractor(x)
        # [B, 512, H/32, W/32]

        batch_size, num_channels, height, width = visual_feats.size()
        # BCHW

        visual_feats = visual_feats.view(batch_size, num_channels * height, width)
        # [B, C*H, W]

        visual_feats = visual_feats.permute(2, 0, 1)
        # [W, B, C*H]

        seq = self.map_visual_to_seq(visual_feats)

        lstm_1, _ = self.b_lstm_1(seq)
        lstm_2, _ = self.b_lstm_2(lstm_1)

        output = self.final_dense(lstm_2)
        # [seq_len, B, num_classes]

        return output
