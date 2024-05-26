
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
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)


        dev_predicted_all = []
        dev_targets_all = []

        for epoch in range(10):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            for sentence, tags in training_data_modified:
                model.zero_grad()
                sentence_in = prepare_sequence(sentence, word_to_ix)
                targets = prepare_sequence(tags, tag_to_ix)
                tag_scores = model(sentence_in)
                loss = loss_function(tag_scores, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(tag_scores, 1)
                correct += (predicted == targets).sum().item()
                total += len(targets)

            train_accuracy = correct / total
            avg_loss = total_loss / len(training_data_modified)

            model.eval()
            dev_correct = 0
            dev_total = 0
            with torch.no_grad():
                for dev_sentence, dev_tags in dev_data_modified:
                    dev_inputs = prepare_sequence(dev_sentence, word_to_ix)
                    dev_targets = prepare_sequence(dev_tags, tag_to_ix)
                    dev_tag_scores = model(dev_inputs)
                    _, dev_predicted = torch.max(dev_tag_scores, 1)
                    dev_predicted_all.extend(dev_predicted.tolist())
                    dev_targets_all.extend(dev_targets.tolist())
                    dev_correct += (dev_predicted == dev_targets).sum().item()
                    dev_total += len(dev_targets)
                
            dev_accuracy = dev_correct / dev_total

        # Calculate test accuracy
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for test_sentence, test_tags in test_data:
                test_sentence = [word if word in word_to_ix else '<UNK>' for word in test_sentence]
                test_inputs = prepare_sequence(test_sentence, word_to_ix)
                test_targets = prepare_sequence(test_tags, tag_to_ix)
                test_tag_scores = model(test_inputs)
                _, test_predicted = torch.max(test_tag_scores, 1)
                test_correct += (test_predicted == test_targets).sum().item()
                test_total += len(test_targets)
        test_accuracy = test_correct / test_total


       
        sentence = input_sent.lower().split()
        sentence_in = prepare_sequence(sentence, word_to_ix)

        model.eval()
        with torch.no_grad():
            tag_scores = model(sentence_in)

        _, predicted_tags_idx = torch.max(tag_scores, 1)
        predicted_tags = [idx_to_tag[idx.item()] for idx in predicted_tags_idx]

        print("Predicted POS Tags:", predicted_tags)

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

        class POSDataset(Dataset):
            def __init__(self, sentences, tags, word_to_ix, tag_to_ix, p, s, embedding_matrix):
                self.sentences = sentences
                self.tags = tags
                self.word_to_ix = word_to_ix
                self.tag_to_ix = tag_to_ix
                self.p = p
                self.s = s
                self.embedding_matrix = embedding_matrix

            def __len__(self):
                return sum(len(sentence) for sentence in self.sentences)

            def __getitem__(self, idx):
                sentence_idx, token_idx = 0, idx
                for sentence in self.sentences:
                    if token_idx < len(sentence):
                        break
                    token_idx -= len(sentence)
                    sentence_idx += 1

                context_window_indices = []
                for i in range(token_idx - self.p, token_idx + self.s + 1):
                    if i < 0 or i >= len(self.sentences[sentence_idx]):  
                        context_window_indices.append(self.word_to_ix[PAD])  
                    else:
                        word = self.sentences[sentence_idx][i]
                        context_window_indices.append(self.word_to_ix.get

        (word, self.word_to_ix[UNK]))  

                context_indices_tensor = torch.tensor(context_window_indices, dtype=torch.long).to(device)
                context_embeddings = self.embedding_matrix[context_indices_tensor].view(-1)

                tag = self.tags[sentence_idx][token_idx] if token_idx < len(self.tags[sentence_idx]) else UNK
                tag_idx = self.tag_to_ix.get(tag, self.tag_to_ix[UNK])

                return context_embeddings, tag_idx


        p = 1
        s = 1

        dataset_tr = POSDataset(train_sentences, train_pos_tags, word_to_ix, tag_to_ix, p, s, embedding_matrix=embedding_matrix)
        dataset_dev = POSDataset(dev_sentences, dev_pos_tags, word_to_ix, tag_to_ix, p, s, embedding_matrix = embedding_matrix)

        def collate_batch(batch):
            inputs, targets = zip(*batch)
            padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
            return padded_inputs, torch.tensor(targets)

        train_dataloader = DataLoader(dataset_tr, batch_size=32, shuffle=True, collate_fn=collate_batch)
        dev_dataloader = DataLoader(dataset_dev, batch_size=32, shuffle=False, collate_fn=collate_batch)

        import torch


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

            


        def train(model, train_dataloader, dev_loader, criterion, optimizer, device, num_epochs=10):
            model.train()
            for epoch in range(num_epochs):
                running_loss = 0.0
                for inputs, targets in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_dataloader)}")
                evaluate(model, dev_dataloader, criterion, device)

        def evaluate(model, dev_dataloader, criterion, device):
            model.eval()
            total_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in tqdm(dev_dataloader, desc="Evaluation", unit="batch"):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)
            accuracy = correct / total
            print(f"Validation Loss: {total_loss / len(dev_dataloader)}, Accuracy: {accuracy}")



        input_dim = embedding_dim * (p + s + 1)  
        hidden_dim = 512
        hidden_dim2 = 128
        output_dim = len(tag_to_ix)  

        model = FFN_POS_Tagger(input_dim, hidden_dim, hidden_dim2, output_dim, activation='ReLU').to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train(model, train_dataloader, dev_dataloader, criterion, optimizer, device)

        def calculate_accuracy_ffnn(model, dataloader, device):
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in dataloader:
                    inputs, targets = inputs.to(device), targets.view(-1).to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)  
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)  
            accuracy = correct / total
            return accuracy

        dev_accuracy = calculate_accuracy_ffnn(model, dev_dataloader, device)
        print(f'Dev Set Accuracy: {dev_accuracy}')

        def test(model, test_loader, criterion, device):
            model.eval()
            total_loss = 0.0
            correct = 0
            total = 0
            target_tags=[]
            predicted_tags=[]
            with torch.no_grad():
                for inputs, targets in tqdm(test_loader, desc="Testing", unit="batch"):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)

            accuracy = correct / total
            print(f"Test Loss: {total_loss / len(test_loader)}, Accuracy: {accuracy}")
            

        dataset_test = POSDataset(test_sentences, test_pos_tags, word_to_ix, tag_to_ix, p, s, embedding_matrix = embedding_matrix)
        test_dataloader = DataLoader(dataset_test, batch_size=32, shuffle=False, collate_fn=collate_batch)

        test(model, test_dataloader, criterion, device)


        context_window_sizes = [0, 1, 2, 3, 4]
        dev_accuracies = [0.9653856277967886, 0.9847328244274809, 0.9828902342721769, 0.9780205317188734, 0.9774940773887866]

        plt.figure(figsize=(8, 6))
        plt.plot(context_window_sizes, dev_accuracies, marker='o', linestyle='-', color='b')
        plt.title('Dev Accuracy vs. Context Window Size')
        plt.xlabel('Context Window Size')
        plt.ylabel('Dev Accuracy')
        plt.grid(True)
        plt.show()


        def evaluate_performance(model, dataloader, device):
            model.eval()
            all_targets = []
            all_predictions = []
            with torch.no_grad():
                for inputs, targets in tqdm(dataloader, desc="Evaluating", unit="batch"):
                    inputs, targets = inputs.to(device), targets.view(-1).to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, -1)
                    all_targets.extend(targets.cpu().numpy())
                    all_predictions.extend(predicted.view(-1).cpu().numpy())

            accuracy = accuracy_score(all_targets, all_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')

            return accuracy, precision, recall, f1


        print('Evaluation Metrics on train set')
        traccuracy, trprecision, trrecall, trf1 = evaluate_performance(model, train_dataloader, device)
        print(f"Accuracy: {traccuracy}")
        print(f"Precision: {trprecision}")
        print(f"Recall: {trrecall}")
        print(f"F1 Score: {trf1}")

        print('Evaluation Metrics on dev set')
        daccuracy, dprecision, drecall, df1 = evaluate_performance(model, dev_dataloader, device)
        print(f"Accuracy: {daccuracy}")
        print(f"Precision: {dprecision}")
        print(f"Recall: {drecall}")
        print(f"F1 Score: {df1}")

        print('Evaluation Metrics on test set')
        taccuracy, tprecision, trecall, tf1 = evaluate_performance(model, test_dataloader, device)
        print(f"Accuracy: {taccuracy}")
        print(f"Precision: {tprecision}")
        print(f"Recall: {trecall}")
        print(f"F1 Score: {tf1}")


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
