# coding=utf-8
import paddlepalm as palm
import json
from paddlepalm.distribute import gpu_dev_count
import argparse
import os
from paras import constantParas
from sklearn.model_selection import StratifiedKFold
    
import pandas as pd
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Machine Reading Comprehension")
    parser.add_argument('--model-name', type=str, default='RoBERTa-zh-large')
    parser.add_argument('--config-name', type=str, default='roberta')
    args = parser.parse_args()
    model_name = args.model_name
    config_name = args.config_name

    max_seqlen = constantParas.max_seqlen[0]
    batch_size = constantParas.batch_size[0]
    num_epochs = constantParas.num_epochs[0]
    lr = constantParas.lr[0]
    weight_decay = constantParas.weight_decay[0]
    num_classes = constantParas.num_classes[0]
    random_seed = constantParas.random_seed[0]
    dropout_prob = constantParas.dropout_prob[0]
    
    print(max_seqlen, batch_size, num_epochs,
    lr, weight_decay, num_classes,random_seed, dropout_prob)
    print_steps = 50

    train = pd.read_csv('./data/train.tsv', sep='\t')   #已经由json文件转化成tsv文件的train
    test = pd.read_csv('./data/test.tsv', sep='\t')     #已经由json文件转化成tsv文件的test
    dev = pd.read_csv('./data/dev.tsv', sep='\t')       ##已经由json文件转化成tsv文件的dev
    all_data = train.append(dev)
    all_data = all_data.reset_index().drop('index', axis=1)

    skf = StratifiedKFold(n_splits=15, random_state=0, shuffle=True)

    for i, (train_index, valid_index) in enumerate(skf.split(all_data, all_data.label)):
        if not os.path.exists('./data/15_fold/{}th/'.format(i)):   #分成15折
            if not os.path.exists('./data/15_fold/'):
                os.mkdir('./data/15_fold/')
            os.mkdir('./data/15_fold/{}th/'.format(i))
        all_data.iloc[train_index].to_csv('./data/15_fold/{}th/'.format(i) + 'train.tsv', sep='\t', index=0)
        all_data.iloc[valid_index].to_csv('./data/15_fold/{}th/'.format(i) + 'dev.tsv', sep='\t', index=0)


    for i in range(0,15):
        save_type = 'ckpt'
        save_path = './outputs/{}_{}fold_model'.format(model_name, i)

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        pre_params = './pretrain_models/pretrain/{}/params'.format(model_name)
        task_name = 'Machine Reading Comprehension'

        vocab_path = './pretrain_models/pretrain/{}/vocab.txt'.format(model_name)
        train_file = './data/15_fold/{}th/train.tsv'.format(i)
        config = json.load(open('./pretrain_models/pretrain/{}/{}_config.json'.format(model_name, config_name)))
        input_dim = config['hidden_size']

        # -----------------------  training -----------------------
        match_reader = palm.reader.MatchReader(vocab_path, max_seqlen, seed=random_seed)
        match_reader.load_data(train_file, file_format='tsv', num_epochs=num_epochs, batch_size=batch_size)
        ernie = palm.backbone.BERT.from_config(config)
        match_reader.register_with(ernie)
        match_head = palm.head.Match(num_classes, input_dim, dropout_prob)
        trainer = palm.Trainer(task_name)
        loss_var = trainer.build_forward(ernie, match_head)
        n_steps = match_reader.num_examples * num_epochs // batch_size
        warmup_steps = int(0.1 * n_steps)
        print('total_steps: {}'.format(n_steps))
        print('warmup_steps: {}'.format(warmup_steps))
        sched = palm.lr_sched.TriangularSchedualer(warmup_steps, n_steps)
        adam = palm.optimizer.Adam(loss_var, lr, sched)
        trainer.build_backward(optimizer=adam, weight_decay=weight_decay)
        trainer.fit_reader(match_reader)
        trainer.load_pretrain(pre_params, False)
        save_steps = num_epochs*match_reader.num_examples // batch_size - 1
        print('===================={}steps 保存==============='.format(save_steps))
        trainer.set_saver(save_path=save_path, save_steps=save_steps, save_type=save_type)
        trainer.train(print_steps=print_steps)