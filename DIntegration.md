# Framework Integration# 
### 2021 
### PJ 
### Hu Chengwei, Shi Yucheng, Jin Haoliang

# Overview
DL is growing rapidly. This project aims to tidy framework and speed ​​up the experiment of the paper.
在这样一套流程之中，最重要的就是数据流。由于数据形式的异构造成了编写代码的多种多样。我们期望能从数据流中总结出相应的范式，达到加快整个项目处理的目的。

# Pytorch Integration
## 参数加载
### 静态参数处理stat_param
1. 配置文件我就要放在json文件中。
2. 配置文件是一段代码。
3. 配置文件来源于configparser。
### 动态参数处理dyn_param
运行期间才生成需要的参数，如：
1. 随机数
## 数据加载器


    DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
               batch_sampler=None, num_workers=0, collate_fn=None,
               pin_memory=False, drop_last=False, timeout=0,
               worker_init_fn=None)


### 加速模型之pinned memory内存

    对CUDA架构而言，主机端的内存可分为两种，一种是pageable memory，即可分页内存；另一种是pinned memory，即页锁定内存。
    
    主机默认分配的是pageable memory，也就是说，根据操作系统的指示，主机虚拟内存（内存空间很小，所以内存只放部分数据，其余不重要的放在硬盘中，看成虚拟内存。）中的数据会对应到不同的物理位置。于是就会产生一种错觉，那就是虚拟内存具有很大的空间。
    
    然而GPU无法安全地访问pageable memory中的数据，因为它无法控制操作系统去移动这些物理数据，所以当我们要把数据从主机的虚拟内存移动到显卡设备的显存中去时，CUDA驱动程序会首先分配临时的pinned memory，然后将主机数据复制并固定到临时pinned memory，由于此时GPU是知道pinned memory的物理内存地址的，所以它可以通过直接内存访问（Direct Memory Access，DMA）在主机和GPU之间快速复制数据。
    
    先比pageable memory，pinned memory的访问速度更快，实验对比，pinned memory的访问时间是前者的一半（0.0167043对比0.0214238）；但缺点就是比pageable memory更消耗内存。使用pinned memory就会失去虚拟内存的功能，尤其是在应用程序中使用每个页锁定内存时都需要分配物理内存,而且这些内存不能交换到磁盘上。这将会导致系统内存会很快的被耗尽，因此应用程序在物理内存较少的机器上会运行失败，不仅如此，还会影响系统上其他应用程序的性能。
    
    所以pytorch考虑到一般情况下内存都不够放数据的，所以的pinnned memory都默认设置为False。

### 预处理位置

    #声明argparse对象
    parser = argparse.ArgumentParser()
 
    #添加命令行参数（required=True的参数必须在命令行设置，其余参数如果不在命令行设置就是用默认值，也可以在命令行设置覆盖默认值）
    ##必须设置的参数
    #处理好的数据集路径 .csv文件所在k路径
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    #所使用的模型 目前支持bert、bert_cnn、bert_lstm、bert_gru、xlnet、xlnet_gru、xlnet_lstm、albert
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    #下载的预训练模型相关文件所在路径(.bin模型参数结构文件,.json模型配置文件，vocab.txt词表文件)
    #注意每种预训练模型都有对应的文件(对应及下载方法 文本分类(三)系列第一篇博客有介绍)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list:")
    #任务名字 THUCNews
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    #模型的预测和checkpoints文件的写入路径
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
 
    ##非必须参数
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    #输入序列最大长度 （需要对batch中所有的序列进行填充 统一成一个长度）
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    #训练、验证和预测
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run predict on the test set.")
    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    #训练阶段和验证阶段 每个gpu上的batch_size
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
                        
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    #初始学习率
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    #正则化系数
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    #epoch数 一个epoch完整遍历一遍数据集
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
 
    #每n次更新(每n个batch) 保存一下日志(损失、准确率等信息)
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    
    #每n次更新(每n个batch) 保存一次参数
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    #初始化模型时 使用的随机种子
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
 
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
 
    #Additional layer parameters
    #预训练模型后加TextCNN     TextCNN相关的参数
    #不同大小卷积核的数量
    parser.add_argument('--filter_num', type=int, default=256, help='number of each size of filter')
    #不同大小卷积核的尺寸
    parser.add_argument('--filter_sizes', type=str, default='3,4,5', help='comma-separated filter sizes to use for convolution')
 
    #预训练模型后加LSTM     LSTM相关的参数
    #隐藏单元个数
    parser.add_argument("--lstm_hidden_size", default=300, type=int,
                        help="")
    #lstm层数
    parser.add_argument("--lstm_layers", default=2, type=int,
                        help="")
    #lstm dropout 丢弃率
    parser.add_argument("--lstm_dropout", default=0.5, type=float,
                        help="") 
 
    ##预训练模型后加GRU     GRU相关的参数
    #隐藏单元个数
    parser.add_argument("--gru_hidden_size", default=300, type=int,
                        help="")
    #gru层数
    parser.add_argument("--gru_layers", default=2, type=int,
                        help="")
    #gru dropout 丢弃率
    parser.add_argument("--gru_dropout", default=0.5, type=float,
                        help="") 
    #解析参数
    args = parser.parse_args()
 
    #把不同大小卷积核的尺寸转换为整数 对原有参数进行覆盖
    args.filter_sizes = [int(size) for size in str(args.filter_sizes).split(',')]


    def convert_examples_to_features(examples, tokenizer,
                                      max_length=512,
                                      task=None,
                                      label_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):
    #把InputExamples对象 转换为输入特征InputFeatures
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """
    is_tf_dataset = False #是否为tensorflow形式数据集
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True
 
    if task is not None:
        processor = processors[task]() #获取自定义任务处理器
        #获取标签列表和输出模式
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))
    
    #类别标签 转换为数字索引
    label_map = {label: i for i, label in enumerate(label_list)}
 
    features = []
    for (ex_index, example) in enumerate(examples): #遍历每一个InputExamples对象(每一条数据)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)
 
        #inputs: dict 调用切分工具中的函数 对每个InputExamples对象进行处理 返回一个字典
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length, #如果序列长于最大长度(统一设置的长度)，截断，其他维持原样
        )
        #input_ids: 输入数据token在词汇表中的索引
        #token_type_ids: 分段token索引，类似segment embedding（对于句子对任务 属于句子A的token为0，句子B的token为1，对于分类任务，只有一个输入句子 全为0）
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        real_token_len = len(input_ids) #输入序列实际长度 不包含填充
 
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        #非填充部分的token对应1
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
 
        # Zero-pad up to the sequence length. 填充部分的长度 >=0
        padding_length = max_length - len(input_ids)
        if pad_on_left: #在输入序列左端填充
            input_ids = ([pad_token] * padding_length) + input_ids
            #填充部分的token 对应0  只对非填充部分的token计算注意力
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:#在输入序列右端填充
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
 
        #填充后 输入序列、attention_mask、token_type_ids的长度都等于max_length（统一设置的长度）
        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)
 
        if output_mode == "classification": #分类任务 把类别标签 转换为索引
            label = label_map[example.label]    #label => index
        elif output_mode == "regression": #回归任务 把标签转换为浮点数
            label = float(example.label)
        else:
            raise KeyError(output_mode)
 
 
        if ex_index < 5: #前5个样本 打印处理效果
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("real_token_len: %s" % (real_token_len))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))
 
        #把构造好的样本 转换为InputFeatures对象 添加到features列表中
        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=label,
                              real_token_len=real_token_len))
 
    if is_tf_available() and is_tf_dataset: #TF 格式的数据
        def gen():
            for ex in features:
                yield ({'input_ids': ex.input_ids,
                         'attention_mask': ex.attention_mask,
                         'token_type_ids': ex.token_type_ids},
                        ex.label)
 
        return tf.data.Dataset.from_generator(gen,
            ({'input_ids': tf.int32,
              'attention_mask': tf.int32,
              'token_type_ids': tf.int32},
             tf.int64),
            ({'input_ids': tf.TensorShape([None]),
              'attention_mask': tf.TensorShape([None]),
              'token_type_ids': tf.TensorShape([None])},
             tf.TensorShape([])))
 
    return features



    def load_and_cache_examples(args, task, tokenizer, evaluate=False, predict=False):
    '''
        将dataset转换为features，并保存在目录cached_features_file中。
    
    args:
        evaluate: False. 若为True，则对dev.csv进行转换
        predict: False. 若为True，则对test.csv进行转换
    return:
        dataset
    '''
 
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
 
    processor = processors[task]()  #THUNewsProcessor() 自定义的任务处理器
    output_mode = output_modes[task]    #classification
 
    # Load data features from cache or dataset file
    #cached_features_file 为数据集构造的特征的保存目录
    if evaluate:
        exec_model = 'dev'
    elif predict:
        exec_model = 'test'
    else:
        exec_model = 'train'
    
    #特征保存目录的命名
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        exec_model,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    
    #如果对数据集已经构造好特征了 直接加载 避免重复处理
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s\n", cached_features_file)
        features = torch.load(cached_features_file)
    #否则对数据集进行处理 得到特征
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels() #标签列表
        #对验证集、测试集、训练集进行处理 把数据转换为InputExample对象
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir)
        elif predict:
            examples = processor.get_test_examples(args.data_dir)
        else:
            examples = processor.get_train_examples(args.data_dir)
        #转换为特征
        #注意xlnet系列模型的数据预处理方式 和 bert系列稍有不同
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        #把数据处理为InputFeatures对象后 存储为cached_features_file
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
 
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
 
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    #构建dataset对象
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

### 数据再处理（转tensor）
## 日志

    可以设置后台运行以及将输出保存在日志文件中。
    nohup python -u sample.py > zdz.log 2>&1 &
    如果使用GPU(后台)运行程序的话,命令：
    CUDA_VISIBLE_DEVICES=2 nohup python -u train.py > zdz.log 2>&1 &
    CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u train.py > zdz.log 2>&1 & #多GPU并行/数据并行


## 并行化
### 单机多进程
### 单机多卡

    # Setup CUDA, GPU & distributed training（单机多卡/多机分布式）
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count() #gpu数量
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

### 分布式
### 数据加载器、取样器DistributedSampler、模型并行化torch.nn.parallel.DistributedDataParallel

    class THUNewsProcessor(DataProcessor):
        """Processor for the SST-2 data set (GLUE version)."""
     
        def get_example_from_tensor_dict(self, tensor_dict):
            """See base class."""
            return InputExample(tensor_dict['idx'].numpy(),
                                tensor_dict['sentence'].numpy().decode('utf-8'),
                                None,
                                str(tensor_dict['label'].numpy()))
     
        #获取训练集、验证集、测试集样本
        #训练集、验证集、测试集提前处理为.csv格式   格式：类别名称,文本
        #通过父类DataProcessor中的_read_csv函数 把csv文件读取为列表 列表中的每一个元素为csv文件的每一行
        def get_train_examples(self, data_dir):
            """See base class."""
            return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")
     
        def get_dev_examples(self, data_dir):
            """See base class."""
            return self._create_examples(self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")
     
        def get_test_examples(self, data_dir):
            """See base class."""
            return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")
     
        def get_labels(self):
            """设置当前数据集的标签"""
            return ["体育", "财经", "房产", "家居", "教育"] #使用了其中5个类别
     
        def _create_examples(self, lines, set_type):
            """Creates examples for the training/dev/test sets."""
            examples = []
            for (i, line) in enumerate(lines): #遍历元组列表，即遍历csv文件每一行/每一条数据
                if i == 0:  #跳过表头 labels,text (把数据处理为csv文件时，是保留表头)
                    continue
                guid = "%s-%s" % (set_type, i) #set_type 训练集/验证集/测试集
                if set_type == 'test': #测试集没有类别标签 如果是测试集 line[0]是文本 标签统一设置为体育
                    text_a = line[0]
                    label = '体育'
                else: #验证集和训练集 line[0]是类别标签，line[1]是文本
                    label = line[0]
                    text_a = line[1]
                    #如有两段文本, 也可以设置text_b 句子对任务(问答)
                #把每条数据 转换为InputExample对象
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
     
            return examples#保存在examples列表中
     
     
     
    tasks_num_labels = {      #分类任务的类别数 存储为字典。键名为任务名(小写)，值为类别数
        "thunews": 5,
    }
     
    processors = {    #任务处理器 存储为字典。键名为任务名(小写)，值为处理器类名(自定义)
        "thunews": THUNewsProcessor,
    }
     
    output_modes = { #输出模式 存储为字典。键名为任务名(小写)，值为classification。分类任务
        "thunews": "classification",
    }


## 训练
### 训练过程


    #通过MODEL_CLASSES字典 传入model_type，得到相应模型的参数配置类、模型类和切分工具类
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    
    #获取并加载预训练模型
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None,
                                        args=args)

    def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
 
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu) #得到训练时的batch大小
    #定义采样方式
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    #构建dataloader
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
 
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
 
    # Prepare optimizer and schedule (linear warmup and decay)
    #以下参数 不使用正则化
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    #定义优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
 
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1: #单机多卡
        model = torch.nn.DataParallel(model)
 
    # Distributed training (should be after apex fp16 initialization)
    #多机分布式
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
 
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
 
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()#训练模式
            batch = tuple(t.to(args.device) for t in batch) #把输入数据 放到设备上
            #分类任务是单句子 不需要设置token_type_ids 默认全为0 batch[2]
            #定义模型的输入参数 字典形式
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids 这些模型没有token_type_ids
            
            if args.model_type in ['bert_cnn']:
                #inputs['real_token_len'] = batch[4]
                pass
            outputs = model(**inputs) #得到模型输出 使用的是XXForSequenceClassification类 他的第一个返回值是损失
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
 
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps  #每个batch都将loss除以gradient_accumulation_steps
 
            #计算梯度
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
 
            epoch_iterator.set_description("loss {}".format(round(loss.item(), 5)))
 
            tr_loss += loss.item()
            #每gradient_accumulation_steps个batch(把这些batch的梯度求和) 更新一次参数
            if (step + 1) % args.gradient_accumulation_steps == 0:  #过gradient_accumulation_steps后才将梯度清零，不是每次更新/每过一个batch清空一次梯度，即每gradient_accumulation_steps次更新清空一次
                #梯度裁剪
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
 
                #反向传播更新参数
                optimizer.step()
                #更新学习率
                scheduler.step()  # Update learning rate schedule
                #清空梯度
                model.zero_grad()
                global_step += 1
 
                #每logging_steps，进行evaluate
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        #验证
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = 'eval_{}'.format(key)
                            logs[eval_key] = value
 
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs['learning_rate'] = learning_rate_scalar
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss
 
                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{'step': global_step}}))
 
                #每save_steps保存checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint 保存参数
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    #单机多卡和多机分布式 保存参数有所不同，不是直接保存model 需要保存model.moudle
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
 
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
 
    if args.local_rank in [-1, 0]:
        tb_writer.close()
 
    return global_step, tr_loss / global_step

### GPU与CPU搬运
### 优化器
#### LR decay
#### COS
#### NUS老师讲的
### 结果保存与加载

    def evaluate(args, model, tokenizer, prefix=""):
        results = {}
        #构建验证数据集 Dataset对象
        eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True)
     
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        #验证阶段batch大小
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        #定义采样方式
        eval_sampler = SequentialSampler(eval_dataset)
        #构建Dataloader
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
     
        # multi-gpu eval 单机多卡
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
     
        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None    #为预测值
        out_label_ids = None    #为真实标签
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval() #验证模式
            batch = tuple(t.to(args.device) for t in batch) #输入数据 转移到device上
     
            with torch.no_grad(): #关闭梯度计算
                #构建模型输入 字典形式。 token_type_ids为batch[2] 分类任务为单输入句子 默认全为0
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids 没有token_type_ids
                outputs = model(**inputs) #获得模型输出
                tmp_eval_loss, logits = outputs[:2] #模型输出的前两项为loss和logits
     
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
     
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        #preds为模型预测的标签 对预测结果按行取argmax
        #和真实标签计算准确率和f1-score
        result = acc_and_f1(preds, out_label_ids)
        results.update(result)
        
        #把相关指标计算结果 保存
        output_eval_file = os.path.join(args.output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
     
        return results

    def predict(args, model, tokenizer, prefix=""):
    #results = {}
    #构建测试数据集 Dataset对象
    pred_dataset = load_and_cache_examples(args, args.task_name, tokenizer, predict=True)
 
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    #测试阶段batch大小
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    #定义采样方式
    eval_sampler = SequentialSampler(pred_dataset)
    #测试集Dataloader
    eval_dataloader = DataLoader(pred_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
 
    # multi-gpu eval 单机多卡
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
 
    # Eval!
    logger.info("***** Running predict {} *****".format(prefix))
    logger.info("  Num examples = %d", len(pred_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None    #为预测值
    #out_label_ids = None    #为真实标签
    for batch in tqdm(eval_dataloader, desc="Predicting"):
        model.eval() #测试模式
        batch = tuple(t.to(args.device) for t in batch)#把测试数据转移到设备上
 
        with torch.no_grad():#关闭梯度计算
            #构建模型输入 字典形式。 token_type_ids为batch[2] 分类任务为单输入句子 默认全为0
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids 没有token_type_ids
            outputs = model(**inputs) #得到模型输出
            tmp_eval_loss, logits = outputs[:2] #前两项为loss、logits
 
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            #out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            #out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
 
    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1) #得到预测的标签 对预测结果按行取argmax
 
    #把预测的标签 输出为csv文件
    pd.DataFrame(preds).to_csv(os.path.join(args.output_dir, "predicted.csv"), index=False)
    #preds.to_csv(os.path.join(args.output_dir, "predicted.csv"))
    #print(preds)
 
 
    #elif args.output_mode == "regression":
    #    preds = np.squeeze(preds)
    #result = acc_and_f1(preds, out_label_ids)
    #results.update(result)
 
    '''
    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    '''


    
    export CUDA_VISIBLE_DEVICES=0,1,2,3  #支持单机多卡
     
    TASK_NAME="THUNews"   #任务名 当前处理的数据集
     
    python run.py \
      --task_name=$TASK_NAME \
      --model_type=bert_cnn \          #使用的模型类型 bert、bert_cnn、xlnet、xlnet_lstm、albert等等 可在项目中扩展
      --model_name_or_path ./pretrained_models/bert-base-chinese \   #下载的相应模型版本的三个文件存储路径。建立采用这种2级路径命名
      --data_dir ./dataset/THUNews/5_5000 \             #数据集所在路径 csv文件
      --output_dir ./results/THUNews/bert_base_chinese_cnn \ #输出结果所在路径 建立采用这种3级目录方式命名，第一级results表示输出结果，第二级表示所处理的数据集，第三级表示所用的模型，由于bert,bert_cnn等都是用的一个bert模型版本，可以结合model_type和所用模型版本进行区分来命名。
      --do_train \
      --do_eval \
      --do_predict \
      --do_lower_case \
      --max_seq_length=512 \
      --per_gpu_train_batch_size=2 \
      --per_gpu_eval_batch_size=16 \
      --gradient_accumulation_steps=1 \
      --learning_rate=2e-5 \
      --num_train_epochs=1.0 \
      --logging_steps=14923 \
      --save_steps=14923 \
      --overwrite_output_dir \
      --filter_sizes='3,4,5' \
      --filter_num=256 \
      --lstm_layers=1 \
      --lstm_hidden_size=512 \
      --lstm_dropout=0.1 \
      --gru_layers=1 \
      --gru_hidden_size=512 \
      --gru_dropout=0.1 \

### 结果画图


# Tensorflow Integration
## Bert4keras问题

# (Optional) TensorRT\tf-serving in deploying
## 最后根据时间来进行


