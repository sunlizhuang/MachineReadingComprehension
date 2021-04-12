# coding=utf-8
import paddlepalm as palm
import json
from paddlepalm.distribute import gpu_dev_count
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Machine Reading Comprehension")
    parser.add_argument('--model-name', type=str, default='RoBERTa-zh-large')
    parser.add_argument('--config-name', type=str, default='roberta')
    args = parser.parse_args()
    config_name = args.config_name
    max_seqlen = 512
    batch_size = 256
    num_epochs = 5
    lr = 3e-5
    weight_decay = 0.0
    num_classes = 3
    random_seed = 1
    dropout_prob = 0.1
    print_steps = 50
    model_name = args.model_name
    for i in range(15):
        ##目前没有15折的其他折数
        if i>0:
            continue
        save_type = 'ckpt'
        pred_model_path = './outputs/{}_{}fold_model/ckpt.step'.format(model_name, i)+str('11789')  #训练好的

        name='test'
        if name == 'test':
            pred_output = './pre_{}/{}fold/'.format(model_name, i) #将结果存入的文件夹名称
        else:
            pred_output = './outputs/{}_dev_predict'.format(model_name)
        import os
        if not os.path.exists(pred_output):
            os.mkdir(pred_output)
        pre_params = './pretrain_models/pretrain/{}/params'.format(model_name)
        task_name = "Machine Reading Comprehension"

        vocab_path = './pretrain_models/pretrain/{}/vocab.txt'.format(model_name)
        predict_file = './data/{}.tsv'.format(name)
        config = json.load(open('./pretrain_models/pretrain/{}/{}_config.json'.format(model_name, config_name)))
        input_dim = config['hidden_size']
        trainer = palm.Trainer(task_name)
        
        # -----------------------  prediction ----------------------- 
        predict_match_reader = palm.reader.MatchReader(vocab_path, max_seqlen, seed=random_seed, phase='predict')
        predict_match_reader.load_data(predict_file, batch_size)
        pred_ernie = palm.backbone.BERT.from_config(config, phase='predict')
        predict_match_reader.register_with(pred_ernie)
        match_pred_head = palm.head.Match(num_classes, input_dim, phase='predict')
        trainer.build_predict_forward(pred_ernie, match_pred_head)
        pred_ckpt = trainer.load_ckpt(pred_model_path)
        trainer.fit_reader(predict_match_reader, phase='predict')
        trainer.predict(print_steps=print_steps, output_dir=pred_output)