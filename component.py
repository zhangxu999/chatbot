from rasa.nlu.components import Component
import typing
from typing import Any, Optional, Text, Dict

# import pkuseg
import jieba
import jieba.posseg as pseg
from gensim.models import Word2Vec, KeyedVectors
import collections
import torch
import torchtext.vocab as Vocab
import torch.utils.data as Data
from torch import nn
from gensim.models import KeyedVectors
from gensim.summarization.bm25 import BM25
import time
import json
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata

small_vec_model = KeyedVectors.load_word2vec_format("data/small_ailab_embedding.txt")
seg = jieba
seg.cut = seg.lcut
seg.cut('nsjjds')

class BiRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(
            input_size=embed_size,
            hidden_size=num_hiddens,
            num_layers=num_layers,
            bidirectional=True,
        )
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.decoder = nn.Linear(4 * num_hiddens, 8)

    def forward(self, inputs):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列长度(seq_len)作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        embeddings = self.embedding(inputs.permute(1, 0))
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs, _ = self.encoder(embeddings)  # output, (h, c)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs


class IntentionRecognition(Component):
    """A new component"""

    # Defines what attributes the pipeline component will
    # provide when called. The listed attributes
    # should be set by the component on the message object
    # during test and train, e.g.
    # ```message.set("entities", [...])```
    provides = []

    # Which attributes on a message are required by this
    # component. e.g. if requires contains "tokens", than a
    # previous component in the pipeline needs to have "tokens"
    # within the above described `provides` property.
    requires = []

    # Defines the default configuration parameters of a component
    # these values can be overwritten in the pipeline configuration
    # of the model. The component should choose sensible defaults
    # and should be able to create reasonable results with the defaults.
    defaults = {}

    # Defines what language(s) this component can handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None which means it can handle all languages.
    # This is an important feature for backwards compatibility of components.
    language_list = None

    def __init__(self, component_config=None, vocab=None, net=None, base_intents=None):
        super().__init__(component_config)
        self.vocab = vocab
        self.net = net
        self.base_intents = base_intents
        self.seg = seg
        self.small_vec_model = small_vec_model

    def get_tokenized_qa_words(self, data):
        """
        data: list of [string, label]
        """

        def tokenizer(text):
            text = text.lower()
            text = self.seg.cut(text)
            return_text = []

            for words in text:
                #             if words in stopwords:
                #                 continue
                if words in self.small_vec_model:
                    return_text.append(words)
                else:
                    return_text += list(words)
            return return_text

        return [tokenizer(question) for question in data]

    def get_vocab_qa(self, data):
        # tokenized_data = get_tokenized_imdb(data)
        counter = collections.Counter([tk for st in data for tk in st])
        return Vocab.Vocab(counter, min_freq=2)

    def preprocess_qa(self, data, vocab):
        max_l = 25  # 将每条评论通过截断或者补0，使得长度变成500

        def pad(x):
            return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

        # tokenized_data = get_tokenized_imdb(data)
        features = torch.tensor(
            [pad([vocab.stoi[word] for word in words]) for words in data]
        )
        return features

    def load_pretrained_embedding(self, words, pretrained_vocab):
        """从预训练好的vocab中提取出words对应的词向量"""
        embed = torch.zeros(len(words), 200)  # 初始化为0
        oov_count = 0  # out of vocabulary
        for i, word in enumerate(words):
            try:
                # idx = pretrained_vocab.stoi[word]
                embed[i, :] = torch.tensor(list(pretrained_vocab[word]))
            except KeyError:
                oov_count += 0
        if oov_count > 0:
            print("There are %d oov words.")
        return embed

    def evaluate_accuracy(
        self,
        data_iter,
        net,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        acc_sum, n = 0.0, 0
        with torch.no_grad():
            for X, y in data_iter:
                if isinstance(net, torch.nn.Module):
                    net.eval()  # 评估模式, 这会关闭dropout
                    acc_sum += (
                        (net(X.to(device)).argmax(dim=1) == y.to(device))
                        .float()
                        .sum()
                        .cpu()
                        .item()
                    )
                    net.train()  # 改回训练模式
                else:  # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                    if "is_training" in net.__code__.co_varnames:  # 如果有is_training这个参数
                        # 将is_training设置成False
                        acc_sum += (
                            (net(X, is_training=False).argmax(dim=1) == y)
                            .float()
                            .sum()
                            .item()
                        )
                    else:
                        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
                n += y.shape[0]
        return acc_sum / n

    def torch_train(
        self, train_iter, test_iter, net, loss, optimizer, device, num_epochs
    ):
        net = net.to(device)
        print("training on ", device)
        batch_count = 0
        for epoch in range(num_epochs):
            train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
            for X, y in train_iter:
                X = X.to(device)
                y = y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                train_l_sum += l.cpu().item()
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                n += y.shape[0]
                batch_count += 1
            test_acc = self.evaluate_accuracy(test_iter, net)
            print(
                "epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec"
                % (
                    epoch + 1,
                    train_l_sum / batch_count,
                    train_acc_sum / n,
                    test_acc,
                    time.time() - start,
                )
            )

    def process_train_data(self, training_data):

        data = [example.text for example in training_data.training_examples]
        self.base_intents = list(training_data.intents)
        label = [
            self.base_intents.index(example.data["intent"])
            for example in training_data.training_examples
        ]
        tokenized_qa_words = self.get_tokenized_qa_words(data)
        vocab = self.get_vocab_qa(tokenized_qa_words)

        X_train, X_test, y_train, y_test = train_test_split(
            tokenized_qa_words, label, train_size=0.99, random_state=42
        )
        train_data, train_label = (
            self.preprocess_qa(X_train, vocab),
            torch.tensor(y_train),
        )

        test_data, test_label = self.preprocess_qa(X_test, vocab), torch.tensor(y_test)
        batch_size = 64
        train_set = Data.TensorDataset(train_data, train_label)
        test_set = Data.TensorDataset(test_data, test_label)
        train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
        test_iter = Data.DataLoader(test_set, batch_size)
        return train_iter, test_iter, vocab

    def train(self, training_data, cfg, **kwargs):
        """Train this component.

        This is the components chance to train itself provided
        with the training data. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.train`
        of components previous to this one."""
        train_iter, test_iter, vocab = self.process_train_data(training_data)
        self.vocab = vocab
        embed_size, num_hidden, num_layers = 200, 100, 2
        self.net = BiRNN(vocab, embed_size, num_hidden, num_layers)
        self.net.embedding.weight.data.copy_(
            self.load_pretrained_embedding(vocab.itos, self.small_vec_model)
        )
        self.net.embedding.weight.requires_grad = False  # 直接加载预训练好的, 所以不需要更新它
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lr, num_epochs = 0.01, 5
        # 要过滤掉不计算梯度的embedding参数
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.net.parameters()), lr=lr
        )
        loss = nn.CrossEntropyLoss()
        self.torch_train(
            train_iter, test_iter, self.net, loss, optimizer, device, num_epochs
        )
        print("training----end--------")

    def process(self, message, **kwargs):
        """Process an incoming message.

        This is the components chance to process an incoming
        message. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.process`
        of components previous to this one."""
        device = list(self.net.parameters())[0].device
        sentence = message.text
        sentence = torch.tensor(
            [self.vocab.stoi[word] for word in sentence], device=device
        )
        intent_confi = self.net(sentence.view((1, -1))).reshape(-1)
        label = torch.argmax(intent_confi).tolist()
        intent, confidence = self.base_intents[label], intent_confi[label].tolist()
        intent_index = ""
        message.set(
            "intent", {"name": intent, "confidence": confidence}, add_to_output=True
        )

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this component to disk for future loading."""
        with open("torch_models/intention_class", "wb") as f:
            torch.save(self.base_intents, f)

        with open("torch_models/intent_classification.model", "wb") as f:
            torch.save(self.net.state_dict(), f)

        with open("torch_models/vocab.bin", "wb") as f:
            torch.save(self.vocab, f)
        print("persist------------")
        return {
            "intention_model_file": "torch_models/intent_classification.model",
            "vocab": "torch_models/vocab.bin",
        }

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any,
    ) -> "Component":
        """Load this component from file."""

        if cached_component:
            return cached_component
        with open("torch_models/intention_class", "rb") as f:
            base_intents = torch.load(f)
        with open(meta["vocab"], "rb") as f:
            vocab = torch.load(f)

        embed_size, num_hidden, num_layers = 200, 100, 2
        net = BiRNN(vocab, embed_size, num_hidden, num_layers)

        with open(meta["intention_model_file"], "rb") as f:
            net.load_state_dict(torch.load(f))

        return cls(
            component_config=meta, vocab=vocab, net=net, base_intents=base_intents
        )


class FindMostLikeQ(Component):
    """A new component"""

    # Defines what attributes the pipeline component will
    # provide when called. The listed attributes
    # should be set by the component on the message object
    # during test and train, e.g.
    # ```message.set("entities", [...])```
    provides = ["response"]

    # Which attributes on a message are required by this
    # component. e.g. if requires contains "tokens", than a
    # previous component in the pipeline needs to have "tokens"
    # within the above described `provides` property.
    requires = []

    # Defines the default configuration parameters of a component
    # these values can be overwritten in the pipeline configuration
    # of the model. The component should choose sensible defaults
    # and should be able to create reasonable results with the defaults.
    defaults = {}

    # Defines what language(s) this component can handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None which means it can handle all languages.
    # This is an important feature for backwards compatibility of components.
    language_list = None

    def __init__(
        self,
        component_config=None,
        q_a_v=None,
        intents_vec=None,
        questions=None,
        base_intents=None,
    ):
        super().__init__(component_config)
        self.seg = seg
        self.small_vec_model = small_vec_model
        self.q_a_v = q_a_v
        self.intents_vec = intents_vec
        self.questions = questions
        self.base_intents = base_intents
        qa_corpus_df = pd.read_csv("data/qa_corpus.csv")
        self.qa_map = {}
        for idx, row in qa_corpus_df.iterrows():
            q = row.question
            a = row.answer
            self.qa_map[q] = a

    def query(self, sentence):
        words = [w for w in sentence if (w in self.small_vec_model)]
        if not words:
            return np.zeros(self.small_vec_model.vector_size)
        vectors = np.mean([self.small_vec_model[w] for w in words], axis=0)
        return vectors

    def get_tokenized_qa_words(self, data):
        """
        data: list of [string, label]
        """

        def tokenizer(text):
            text = text.lower()
            text = self.seg.cut(text)
            return_text = []

            for words in text:
                #             if words in stopwords:
                #                 continue
                if words in self.small_vec_model:
                    return_text.append(words)
                else:
                    return_text += list(words)
            return return_text

        return [tokenizer(question) for question in data]

    def train(self, training_data, cfg, **kwargs):
        """Train this component.

        This is the components chance to train itself provided
        with the training data. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.train`
        of components previous to this one."""
        """新构造"""
        self.base_intents = list(training_data.intents)
        self.intents_vec = np.array(
            [
                self.base_intents.index(example.data["intent"])
                for example in training_data.training_examples
            ]
        )
        questions = [example.text for example in training_data.training_examples]
        question_vectors = np.array(
            [
                self.query(sentence)
                for sentence in self.get_tokenized_qa_words(questions)
            ]
        )

        self.q_a_v = question_vectors

    def process(self, message, **kwargs):
        """Process an incoming message.

        This is the components chance to process an incoming
        message. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.process`
        of components previous to this one."""
        sentence = message.text
        question_vectors = [
            self.query(sentence) for sentence in self.get_tokenized_qa_words([sentence])
        ]
        intent = message.data["intent"]["name"]
        intent_num = self.base_intents.index(intent)
        intent_index = np.argwhere(self.intents_vec == intent_num).reshape(-1)
        intent_question = self.q_a_v[intent_index]
        sim_mat = cosine_similarity(question_vectors, intent_question)
        top_posi = np.argsort(sim_mat, axis=1)[0][-10:][::-1]
        full_index = intent_index[top_posi]
        sim_value = sim_mat[0, top_posi]
        if "full_index" not in message.data:
            message.set("full_index", full_index.tolist())
        else:
            exist = message.get("full_index")
            full_index = list(set(exist + full_index.tolist()))
            message.set("full_index", full_index)
        sim_value = json.dumps(sim_value.tolist(), ensure_ascii=False)
        message.set("top_similar_value", sim_value, add_to_output=True)

        # ========================================
        sim_mat_part = cosine_similarity(question_vectors, self.q_a_v[full_index])[0]

        candicate_sentences = self.questions[full_index].tolist()
        candicate_sentences.insert(0, sentence)
        sentence_tokens = self.get_tokenized_qa_words(candicate_sentences)
        bm25_object = BM25(sentence_tokens[1:])
        bm25_score = np.array(bm25_object.get_scores(sentence_tokens[0]))
        feature_score = np.array([sim_mat_part, bm25_score]).T
        scaler = MinMaxScaler()
        scaler.fit(feature_score)
        feature_score = scaler.transform(feature_score)
        weight = np.array([0.8, 0.2])
        final_score = (feature_score * weight).sum(axis=1)
        # final_score = sim_mat_part + bm25_score

        best_match_question = self.questions[full_index[np.argmax(final_score)]]
        message.set("best_match", best_match_question, add_to_output=True)
        responses = self.qa_map.get(best_match_question)
        message.set("response", responses, add_to_output=True)
        # word_vec_value = np.sort(sim_mat_part)[::-1]
        # word_vec_index = np.argsort(sim_mat_part)[::-1]
        # message.set("word_vec_value", word_vec_value.tolist(), add_to_output=True)
        # message.set(
        #     "word_vec_qquestion",
        #     self.questions[np.array(full_index)[word_vec_index]].tolist(),
        #     add_to_output=True,
        # )
        # bm25_value = np.sort(bm25_score)[::-1]
        # bm25_index = np.argsort(bm25_score)[::-1]
        # message.set("bm25_value", bm25_value.tolist(), add_to_output=True)
        # message.set(
        #     "bm25_question",
        #     self.questions[np.array(full_index)[bm25_index]].tolist(),
        #     add_to_output=True,
        # )
        # final_score_value = np.sort(final_score)[::-1]
        # final_score_index = np.argsort(final_score)[::-1]
        # message.set("final_score_value", final_score_value.tolist(), add_to_output=True)
        # message.set(
        #     "final_score",
        #     self.questions[np.array(full_index)[final_score_index]].tolist(),
        #     add_to_output=True,
        # )

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this component to disk for future loading."""
        np.save("torch_models/q_a_v.npy", self.q_a_v)
        np.save("torch_models/intents_vec.npy", self.intents_vec)
        with open("torch_models/base_intents", "wb") as f:
            pickle.dump(self.base_intents, f)

        return {
            "q_a_v": "torch_models/q_a_v",
        }

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any,
    ) -> "Component":
        """Load this component from file."""

        if cached_component:
            return cached_component
        q_a_v = np.load("torch_models/q_a_v.npy")
        intents_vec = np.load("torch_models/intents_vec.npy")
        questions = np.load("torch_models/questions.npy")
        with open("torch_models/base_intents", "rb") as f:
            base_intents = pickle.load(f)

        return cls(
            component_config=meta,
            q_a_v=q_a_v,
            intents_vec=intents_vec,
            questions=questions,
            base_intents=base_intents,
        )


class BoolSearch(Component):
    def __init__(
        self, component_config=None, bool_table=None, words_index=None, questions=None
    ):
        super().__init__(component_config)
        self.bool_table = bool_table
        self.words_index = words_index
        self.questions = questions

    def train(self, training_data, cfg, **kwargs):
        corpus = [seg.cut(sen.text) for sen in training_data.training_examples]
        vectorizer = CountVectorizer(analyzer=lambda x: x)
        bool_table = vectorizer.fit_transform(corpus)
        self.bool_table = bool_table.toarray() > 0
        self.words_index = vectorizer.vocabulary_
        self.questions = np.array([sen.text for sen in training_data.training_examples])

    def process(self, message, **kwargs):
        sentence = message.text
        words = [
            w.word
            for w in pseg.cut(sentence)
            if (w.flag not in ["x","w"]) and self.words_index.get(w.word)
        ]
        words_index = [self.words_index[w] for w in words]
        weight_score = self.bool_table[:, words_index].sum(axis=1)

        sentence_index = np.argwhere(weight_score > 0).reshape(-1)
        score = weight_score[sentence_index]
        full_index = sentence_index[np.argsort(score)[::-1][:20]]
        # sentence = self.questions[sentence_index]
        if not hasattr(message, "full_index"):
            message.set("full_index", full_index.tolist())
        else:
            exist = message.get("full_index")
            full_index = list(set(exist + full_index.tolist()))
            message.set("full_index", full_index)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        np.save("torch_models/bool_table.npy", self.bool_table)
        with open("torch_models/word_index.dict", "wb") as f:
            torch.save(self.words_index, f)
        np.save("torch_models/questions.npy", self.questions)

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any,
    ) -> "Component":
        """Load this component from file."""

        if cached_component:
            return cached_component
        bool_table = np.load("torch_models/bool_table.npy")
        with open("torch_models/word_index.dict", "rb") as f:
            words_index = torch.load(f)
        questions = np.load("torch_models/questions.npy")

        return cls(
            component_config=meta,
            bool_table=bool_table,
            words_index=words_index,
            questions=questions,
        )

