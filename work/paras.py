class constantParas():
    
    max_seqlen = 512, ##候选350， 280， 256
    batch_size = 32,
    num_epochs = 5,
    lr = 3e-5,
    weight_decay = 0.0,
    num_classes = 3,
    random_seed = 1,
    dropout_prob = 0.1,

    label_map = {
        1:'Yes',
        2:'Depends',
        0:'No'
    }
