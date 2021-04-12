from paddlepalm import downloader
downloader.ls('pretrain')
downloader.download('pretrain', 'ERNIE-v1-zh-base', './pretrain_models')
downloader.download('pretrain', 'RoBERTa-zh-base', './pretrain_models')
downloader.download('pretrain', 'RoBERTa-zh-large', './pretrain_models')
downloader.download('pretrain', 'BERT-zh-base', './pretrain_models')