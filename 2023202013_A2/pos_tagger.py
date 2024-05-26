
import sys



if __name__ == "__main__":
    model_type = sys.argv[1]  # -f for FFN or -r for RNN
      # Update this path
    input_sent = input("Enter a sentence: ")
    
    if model_type == '-r':
        model_path = 'ffnn_pos_tagger_model.pth'
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import torch.nn.functional as F
        from collections import Counter
        
        def prepare_sequence(seq, to_ix):
            idxs = [to_ix[w] for w in seq]
            return torch.tensor(idxs, dtype=torch.long)

        def load_conllu_data(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = []
                sentence = []
                tags = []
                for line in file:
                    if line.strip() == "":
                        if sentence and tags:
                            data.append((sentence, tags))
                            sentence = []
                            tags = []
                    elif not line.startswith("#"):  # Skip comment lines
                        parts = line.split('\t')
                        if len(parts) > 3:
                            word = parts[1]
                            tag = parts[3]
                            sentence.append(word)
                            tags.append(tag)
                if sentence and tags:
                    data.append((sentence, tags))
            return data


        train_path = 'en_atis-ud-train.conllu'  
        dev_path = 'en_atis-ud-dev.conllu'  
        test_path = 'en_atis-ud-test.conllu'  
        training_data = load_conllu_data(train_path)
        dev_data = load_conllu_data(dev_path)
        test_data = load_conllu_data(test_path)


        tag_to_ix = {'<PAD>': 0, 'CCONJ': 1, 'NOUN': 2, 'AUX': 3, 'PRON': 4, 'DET': 5, 'ADP': 6, 'NUM': 7, 'ADJ': 8, 'ADV': 9, 'PART': 10, 'PROPN': 11, 'INTJ': 12, 'VERB': 13, '<UNK>': 14, 'SYM': 15}
        idx_to_tag = {idx: tag for tag, idx in tag_to_ix.items()}


        def build_vocab(train_data, dev_data):
            word_counts = Counter()

            for sentence, _ in train_data + dev_data:
                word_counts.update(sentence)

            unk_count = 0
            for word, count in list(word_counts.items()):
                if count < 3:
                    unk_count += count
                    del word_counts[word]

            word_counts['<UNK>'] = unk_count

            word_to_ix = {word: idx for idx, word in enumerate(word_counts)}

            return word_to_ix, word_counts

        word_to_ix, word_counts = build_vocab(training_data, dev_data)

        training_data_modified = []
        for sentence, tags in training_data:
            modified_sentence = [word if word_counts[word] >= 3 else '<UNK>' for word in sentence]
            training_data_modified.append((modified_sentence, tags))

        dev_data_modified = []
        for sentence, tags in dev_data:
            modified_sentence = [word if word_counts[word] >= 3 else '<UNK>' for word in sentence]
            dev_data_modified.append((modified_sentence, tags))

        EMBEDDING_DIM = 100
        HIDDEN_DIM = 128

        class LSTMTagger(nn.Module):
            def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
                super(LSTMTagger, self).__init__()
                self.hidden_dim = hidden_dim
                self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
                self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

            def forward(self, sentence):
                embeds = self.word_embeddings(sentence)
                lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
                tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
                tag_scores = F.log_softmax(tag_space, dim=1)
                return tag_scores

        model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))

        model.load_state_dict(torch.load('LSTM_model.pth'))

        def predict_pos_tags():
        
            def prepare_sequencei(seq, to_ix):
                idxs = [to_ix.get(w, to_ix['<UNK>']) for w in seq]  # Use '<UNK>' for unknown words
                return torch.tensor(idxs, dtype=torch.long)
                
            sentence = input_sent.lower().split()
            sentence_in = prepare_sequencei(sentence, word_to_ix)

            model.eval()
            with torch.no_grad():
                tag_scores = model(sentence_in)

            _, predicted_tags_idx = torch.max(tag_scores, 1)
            predicted_tags = [idx_to_tag[idx.item()] for idx in predicted_tags_idx]

            print("Predicted POS Tags:", predicted_tags)
        
        predict_pos_tags()

    elif model_type == '-f':

        from conllu import parse_incr
        from collections import Counter, defaultdict
        import torch
        from torch.utils.data import Dataset, DataLoader
        from torch.nn.utils.rnn import pad_sequence
        import torchtext.vocab as vocab
        import torch.nn as nn
        import torch.optim as optim
        import matplotlib.pyplot as plt
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        from tqdm import tqdm


        # File paths
        train_file_path = 'en_atis-ud-train.conllu'
        dev_file_path = 'en_atis-ud-dev.conllu'
        test_file_path = 'en_atis-ud-test.conllu'


        UNK = "<UNK>"
        PAD = "<PAD>"
        START = "<START>"
        END = "<END>"

        def load_and_preprocess_conllu(file_path):
            sentences = []
            pos_tags = []

            with open(file_path, 'r', encoding='utf-8') as file:
                for tokenlist in parse_incr(file):
                    sentence = []
                    tags = []
                    for token in tokenlist:
                        if token['form'].isalpha():
                            sentence.append(token['form'].lower())
                            tags.append(token['upos'])
                    sentences.append([START] + sentence + [END])
                    pos_tags.append([PAD] + tags + [PAD])
            return sentences, pos_tags

        # Load and preprocess the datasets
        train_sentences, train_pos_tags = load_and_preprocess_conllu(train_file_path)
        dev_sentences, dev_pos_tags = load_and_preprocess_conllu(dev_file_path)
        test_sentences, test_pos_tags = load_and_preprocess_conllu(test_file_path)

        all_sentences = train_sentences + dev_sentences

        word_freq = Counter([word for sentence in all_sentences for word in sentence])

        def replace_low_freq_words(sentences, word_freq, threshold=2):
            processed_sentences = []
            for sentence in sentences:
                processed_sentence = [word if word_freq[word] >= threshold else UNK for word in sentence]
                processed_sentences.append(processed_sentence)
            return processed_sentences

        train_sentences = replace_low_freq_words(train_sentences, word_freq)
        dev_sentences = replace_low_freq_words(dev_sentences, word_freq)

        # Path to the directory where you extracted the GloVe embeddings
        glove_path = 'embeddings/glove.6B'

        # Load 100-dimensional GloVe embeddings
        glove = vocab.GloVe(name='6B', dim=100, cache=glove_path)

        embedding_dim = 100
        word_to_ix = {word: index + 1 for index, word in enumerate(glove.itos)}  
        word_to_ix[UNK] = len(word_to_ix) + 1
        word_to_ix[PAD] = len(word_to_ix) + 1
        word_to_ix[START] = len(word_to_ix) + 1
        word_to_ix[END] = len(word_to_ix) + 1

        embedding_matrix = torch.zeros((len(word_to_ix) + 1, embedding_dim))  
        for word, idx in word_to_ix.items():
            if word in glove.stoi:
                embedding_matrix[idx] = glove.vectors[glove.stoi[word]]
            else:  
                embedding_matrix[idx] = torch.randn(embedding_dim)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        embedding_matrix = embedding_matrix.to(device)  

        pos_tags = set([tag for sentence in train_pos_tags for tag in sentence])

        tag_to_ix = {'<PAD>': 0, 'CCONJ': 1, 'NOUN': 2, 'AUX': 3, 'PRON': 4, 'DET': 5, 'ADP': 6, 'NUM': 7, 'ADJ': 8, 'ADV': 9, 'PART': 10, 'PROPN': 11, 'INTJ': 12, 'VERB': 13, '<UNK>': 14}

        p = 1
        s = 1


        class FFN_POS_Tagger(nn.Module):
            def __init__(self, input_dim, hidden_dim, hidden_dim2, output_dim, activation):
                super(FFN_POS_Tagger, self).__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.output_dim = output_dim
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.activation1 = self.get_activation(activation)  
                self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
                self.activation2 = self.get_activation(activation)  
                self.fc3 = nn.Linear(hidden_dim2, output_dim)

            def get_activation(self, activation):
                if activation == 'ReLU':
                    return nn.ReLU()
                elif activation == 'Tanh':
                    return nn.Tanh()
                else:
                    raise ValueError("Invalid activation function. Choose 'ReLU' or 'Tanh'.")

            def forward(self, x):
                batch_size = x.size(0)  
                x = x.view(batch_size, -1)  
                x = self.fc1(x)
                x = self.activation1(x)
                x = self.fc2(x)
                x = self.activation2(x)
                x = self.fc3(x)
                return x


        input_dim = embedding_dim * (p + s + 1)  
        hidden_dim = 512
        hidden_dim2 = 128
        output_dim = len(tag_to_ix)  

        model = FFN_POS_Tagger(input_dim, hidden_dim, hidden_dim2, output_dim, activation='ReLU').to(device)
        model.load_state_dict(torch.load('ffnn_pos_tagger_model.pth'))


        def preprocess_and_predict(sentence, word_to_ix, tag_to_ix, model, device, embedding_matrix, p, s):
            tokens = sentence.lower().split()
            processed_sentence = [PAD] + tokens + [PAD]
            predictions = []

            model.eval()
            with torch.no_grad():
                for i, token in enumerate(tokens):
                    context_window = processed_sentence[max(0, i):i+p+1] + processed_sentence[i+1:min(i+1+s, len(processed_sentence))]
                    context_indices = [word_to_ix.get(word, word_to_ix['<UNK>']) for word in context_window]

                    while len(context_indices) < p + s + 1:
                        context_indices.append(word_to_ix['<PAD>'])  
                    context_tensor = torch.tensor([context_indices], dtype=torch.long).to(device)
                    context_embeddings = embedding_matrix[context_tensor].view(1, -1)
                    output = model(context_embeddings)
                    predicted_tag_ix = output.argmax(1).item()
                    predictions.append((token, list(tag_to_ix.keys())[list(tag_to_ix.values()).index(predicted_tag_ix)]))

            return predictions

        predicted_tags = preprocess_and_predict(input_sent, word_to_ix, tag_to_ix, model, device, embedding_matrix, p, s)
        for token, tag in predicted_tags:
            print(f"{token} : {tag}")

    else:
        raise ValueError("Invalid model type specified. Use -f for FFN or -r for RNN.")
