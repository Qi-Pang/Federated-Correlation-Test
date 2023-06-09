import re
import xml.sax.saxutils as saxutils
from bs4 import BeautifulSoup

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd
from pandas import DataFrame
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from random import random
import numpy as np
from tqdm import tqdm
import argparse
from scipy.stats import levy_stable
from scipy.special import gamma
from multiprocessing import Pool
import copy
import csv
import pickle

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from torchvision import transforms as T

def prepare_data():
    data_folder = '../dataset/reuters21578/'
    # sgml_number_of_files = 17
    # sgml_number_of_files = 5
    sgml_number_of_files = 22
    sgml_file_name_template = 'reut2-{}.sgm'
    category_files = {
        'to_': ('Topics', 'all-topics-strings.lc.txt'),
        'pl_': ('Places', 'all-places-strings.lc.txt'),
        'pe_': ('People', 'all-people-strings.lc.txt'),
        'or_': ('Organizations', 'all-orgs-strings.lc.txt'),
        'ex_': ('Exchanges', 'all-exchanges-strings.lc.txt')
    }

    # Read all categories
    category_data = []

    for category_prefix in category_files.keys():
        with open(data_folder + category_files[category_prefix][1], 'r') as file:
            for category in file.readlines():
                category_data.append([category_prefix + category.strip().lower(), 
                                    category_files[category_prefix][0], 0])

    # Create category dataframe
    news_categories = DataFrame(data=category_data, columns=['Name', 'Type', 'Newslines'])

    # Those are the top 20 categories we will use for the classification
    selected_categories = ['pl_usa', 'to_earn', 'to_acq', 'pl_uk', 'pl_japan', 'pl_canada', 'to_money-fx',
    'to_crude', 'to_grain', 'pl_west-germany', 'to_trade', 'to_interest',
    'pl_france', 'or_ec', 'pl_brazil', 'to_wheat', 'to_ship', 'pl_australia',
    'to_corn', 'pl_china']

    # 10 classes
    # selected_categories = ['pl_usa', 'to_earn', 'to_acq', 'pl_uk', 'pl_japan', 'pl_canada', 'to_money-fx',
    # 'to_crude', 'to_grain', 'pl_west-germany']

    # Parse SGML files
    document_X = []
    document_Y = []

    def strip_tags(text):
        return re.sub('<[^<]+?>', '', text).strip()

    def unescape(text):
        return saxutils.unescape(text)

    def update_frequencies(categories):
        for category in categories:
            idx = news_categories[news_categories.Name == category].index[0]
            f = news_categories._get_value(idx, 'Newslines')
            news_categories.at[idx, 'Newslines'] = f+1

    def to_category_vector(categories, target_categories):
        vector = np.zeros(len(target_categories)).astype(np.float32)
        
        for i in range(len(target_categories)):
            if target_categories[i] in categories:
                vector[i] = 1.0
        
        return vector

    # Iterate all files
    for i in range(sgml_number_of_files):
        file_name = sgml_file_name_template.format(str(i).zfill(3))
        print('Reading file: %s' % file_name)
        
        with open(data_folder + file_name, 'rb') as file:
            content = BeautifulSoup(file.read().lower(), "lxml")
            
            for newsline in content('reuters'):
                document_categories = []
                
                # News-line Id
                # document_id = newsline['newid']
                
                # News-line text
                document_body = strip_tags(str(newsline('text')[0].text)).replace('reuter\n&#3;', '')
                document_body = unescape(document_body)
                
                # News-line categories
                topics = newsline.topics.contents
                places = newsline.places.contents
                people = newsline.people.contents
                orgs = newsline.orgs.contents
                exchanges = newsline.exchanges.contents
                
                for topic in topics:
                    document_categories.append('to_' + strip_tags(str(topic)))
                    
                for place in places:
                    document_categories.append('pl_' + strip_tags(str(place)))
                    
                for person in people:
                    document_categories.append('pe_' + strip_tags(str(person)))
                    
                for org in orgs:
                    document_categories.append('or_' + strip_tags(str(org)))
                    
                for exchange in exchanges:
                    document_categories.append('ex_' + strip_tags(str(exchange)))
                    
                # Create new document    
                update_frequencies(document_categories)
                
                document_X.append(document_body)
                document_Y.append(to_category_vector(document_categories, selected_categories))

    news_categories.sort_values(by='Newslines', ascending=False, inplace=True)
    # Selected categories
    selected_categories = np.array(news_categories["Name"].head(20))
    # num_categories = 20
    # print(news_categories.head(num_categories))
    return document_X, document_Y

def clean_data(doc_X, doc_Y):
    lemmatizer = WordNetLemmatizer()
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    stop_words = set(stopwords.words("english"))

    def cleanUpSentence(r, stop_words = None):
        r = r.lower().replace("<br />", " ")
        r = re.sub(strip_special_chars, "", r.lower())
        if stop_words is not None:
            words = word_tokenize(r)
            filtered_sentence = []
            for w in words:
                w = lemmatizer.lemmatize(w)
                if w not in stop_words:
                    filtered_sentence.append(w)
            return " ".join(filtered_sentence)
        else:
            return r
    
    totalX = []
    totalY = np.array(doc_Y)
    for _, doc in enumerate(doc_X):
        totalX.append(cleanUpSentence(doc, stop_words))

    return totalX, totalY

def convert_words(totalX, totalY):
    xLengths = [len(word_tokenize(x)) for x in totalX]
    h = sorted(xLengths)  #sorted lengths
    maxLength =h[len(h)-1]
    print("max input length is: ",maxLength)
    max_vocab_size = 200000
    input_tokenizer = Tokenizer(max_vocab_size)
    input_tokenizer.fit_on_texts(totalX)
    input_vocab_size = len(input_tokenizer.word_index) + 1
    print("input_vocab_size:",input_vocab_size)
    totalX = np.array(pad_sequences(input_tokenizer.texts_to_sequences(totalX), maxlen=maxLength))
    print(totalY.shape)
    return totalX, totalY

def construct_global_table(totalX, totalY):
    N_doc = totalX.shape[0]
    N_class = np.max(totalY) + 1
    chi_table = dict()
    print('total doc number: ', N_doc)
    for i in tqdm(range(N_doc)):
        terms = list(set(totalX[i]))
        for term in terms:
            if term not in chi_table.keys():
                chi_table[term] = np.zeros((2, N_class))
            chi_table[term][0, totalY[i]] += 1

    labels_index, labels_freq = np.unique(totalY, return_counts=True)
    for label in tqdm(range(N_class)):
        freq = labels_freq[np.where(labels_index == label)]
        for term in chi_table.keys():
            chi_table[term][1, label] = freq - chi_table[term][0, label]

    return chi_table

def get_global_E(chi_table, N_class, N_doc):
    for key in chi_table.keys():
        temp_table = chi_table[key]
        y_sum = np.sum(temp_table, axis=1).reshape(-1 ,1)
        x_sum = np.sum(temp_table, axis=0).reshape(1, -1)
        temp_table = np.matmul(y_sum, x_sum) / N_doc
        assert temp_table.shape == chi_table[key].shape
        chi_table[key] = temp_table
    
    return chi_table

def baseline(totalX, totalY, E_table, upload_table):
    N_doc = totalX.shape[0]
    for i in range(N_doc):
        terms = list(set(totalX[i]))
        for term in terms:
            upload_table[term][0, totalY[i]] += 1

    labels_index, labels_freq = np.unique(totalY, return_counts=True)
    for i in tqdm(range(labels_index.shape[0])):
        label = labels_index[i]
        freq = labels_freq[i]
        for term in E_table.keys():
            upload_table[term][1, label] = freq - upload_table[term][0, label]
    for term in E_table.keys():
        upload_table[term] = np.sum(np.square(np.divide((upload_table[term] - E_table[term]), np.sqrt(E_table[term]))))
    return upload_table

def sample_from_stable(params):
    samples, N_class = params
    rv = levy_stable(2.0, 0.0)
    return rv.rvs(size=[samples, 2, N_class])

def server_gaussian(chi_table, N_class, samples):
    gaussian_table = dict()

    pool = Pool(16)
    pool_result = list(pool.map(sample_from_stable, [(samples, N_class) for key in chi_table.keys()]))

    for index, key in enumerate(chi_table.keys()):
        gaussian_table[key] = pool_result[index]

    return gaussian_table

def client_calculate(params):
    local_dataX, local_dataY, clients_num, samples, N_class = params
    # for key in gaussian_table.keys():
    #     upload_table_sample[key] = np.zeros((samples, 1))
    #     upload_table[key] = np.zeros((2, N_class))
    upload_table_sample = copy.deepcopy(temp_sample)
    upload_table = copy.deepcopy(temp_table)
    # upload_table_sample, upload_table, local_dataX, local_dataY, gaussian_table, E_table, clients_num, samples = params
    local_N_doc = local_dataX.shape[0]

    for i in range(local_N_doc):
        terms = list(set(local_dataX[i]))
        for term in terms:
            upload_table[term][0, local_dataY[i]] += 1

    labels_index, labels_freq = np.unique(local_dataY, return_counts=True)
    for i in range(labels_index.shape[0]):
        label = labels_index[i]
        freq = labels_freq[i]
        for term in chi_table.keys():
            upload_table[term][1, label] = freq - upload_table[term][0, label]
    
    for term in chi_table.keys():
        global_table = chi_table[term]
        upload_table[term] = np.divide((global_table / clients_num - upload_table[term]), np.sqrt(global_table))
        for k in range(samples):
            upload_table_sample[term][k] = np.sum(upload_table[term] * gaussian_table[term][k])

    return upload_table_sample

def geometric_mean(alpha, sketch_size, x):
    return np.prod(np.power(np.abs(x), alpha/sketch_size))/np.power(2*gamma(alpha/sketch_size)*gamma(1-1/sketch_size)*np.sin(np.pi*alpha/2/sketch_size)/np.pi, sketch_size)

def server_agg(samples, clients_table_list, dropout=0):
    nworker = len(clients_table_list)
    results = dict()
    for key in clients_table_list[0].keys():
        results[key] = np.zeros((samples, 1))
    dropout_list = np.random.choice(nworker, int(dropout * nworker // 100), replace=False).tolist()
    for client_id in tqdm(range(nworker)):
        if client_id in dropout_list:
            continue
        for key in clients_table_list[0].keys():
            results[key] += clients_table_list[client_id][key]

    for key in clients_table_list[0].keys():
        results[key] = geometric_mean(2.0, samples, results[key])

    return results

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--nworker', type=int, default=100)
    # parser.add_argument('--row', type=int, default=2)
    # parser.add_argument('--col', type=int, default=2)
    parser.add_argument('--samples', type=int, default=50)
    # implement parallel using pool to accelerate the process
    parser.add_argument('--para_level', type=int, default=1)

    parser.add_argument('--type', type=str, default='filter')
    parser.add_argument('--device', type=int, default=0)

    return parser.parse_args()

def select_features(X, selected_terms, feature_num=40000):
    N_doc = X.shape[0]
    result_data = np.zeros((N_doc, feature_num))
    for i in tqdm(range(N_doc)):
        temp_list = list(set(X[i]))
        for j in range(feature_num):
            if selected_terms[j] in temp_list:
                result_data[i, j] = 1

    return result_data

def select_features_orig(X):
    N_doc = X.shape[0]
    feature_num = np.max(X) + 1
    result_data = np.zeros((N_doc, feature_num))
    for i in tqdm(range(N_doc)):
        temp_list = list(set(X[i]))
        result_data[i, temp_list] = 1
    return result_data

class Reuters_train_dataset(Dataset):
    def __init__(self, arrayX, arrayY):
        self.arrayX = arrayX.astype(np.float32)
        self.arrayY = arrayY
        self.transform = T.Compose([T.ToTensor(),])
    def __len__(self):
        return self.arrayX.shape[0]
    def __getitem__(self, i):
        tempX = self.arrayX[i].reshape(1, -1)
        tempY = self.arrayY[i].reshape(1, -1)
        return self.transform(tempX), self.transform(tempY)

def weight_init(m):
    torch.manual_seed(0)
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

def train_LR(X, Y, mode='fed'):
    torch.manual_seed(0)
    p = np.random.permutation(len(X))
    X = X[p]
    Y = Y[p]
    BATCH_SIZE = 10
    num_classes = 20
    N_doc = X.shape[0]
    train_data = Reuters_train_dataset(X[:int(N_doc * 0.8)], Y[:int(N_doc * 0.8)])
    train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True, num_workers=8)

    test_data = Reuters_train_dataset(X[int(N_doc * 0.8):], Y[int(N_doc * 0.8):])
    test_loader = DataLoader(test_data, 64, shuffle=False, num_workers=8)

    model = torch.nn.Linear(X.shape[1], num_classes).cuda()
    model.apply(weight_init)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    
    for epoch in range(150):
        model.train()
        for ii, (dataX, labelY) in enumerate(train_loader):
            inputX = Variable(dataX).cuda()
            labelY = Variable(labelY).cuda()
            optimizer.zero_grad()
            output = model(inputX)
            loss = criterion(output, labelY)
            loss.backward()
            optimizer.step()
            # if ii % 10 == 0:
            #     print('Loss: {:.3f}'.format(loss.item()))

        model.eval()
        acc = 0
        with torch.no_grad():
            for ii, (dataX, labelY) in enumerate(test_loader):
                test_X = Variable(dataX).cuda()
                test_out = model(test_X)
                test_out = torch.sigmoid(test_out).cpu().detach().numpy()
                test_out = np.round(test_out)
                acc += accuracy_score(test_out.reshape(-1, num_classes), labelY.numpy().astype(int).reshape(-1, num_classes), normalize=False)
        acc = acc / len(test_data)
        temp_str = str(epoch) + '\t' + str(acc) + '\n'
        print('Epoch: ', epoch, 'Acc:', acc)
        filename = mode + '.txt'
        fp = open(filename, 'a')
        fp.write(temp_str)
        fp.close()
        
def train_LR_orig(X, Y):
    p = np.random.permutation(len(X))
    X = X[p]
    Y = Y[p]
    BATCH_SIZE = 10
    num_classes = 20
    N_doc = X.shape[0]
    train_data = Reuters_train_dataset(X[:int(N_doc * 0.8)], Y[:int(N_doc * 0.8)])
    train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True, num_workers=8)

    test_data = Reuters_train_dataset(X[int(N_doc * 0.8):], Y[int(N_doc * 0.8):])
    test_loader = DataLoader(test_data, 64, shuffle=False, num_workers=8)

    model = torch.nn.Linear(X.shape[1], num_classes).cuda()
    model.apply(weight_init)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    
    for epoch in range(150):
        model.train()
        for ii, (dataX, labelY) in enumerate(train_loader):
            inputX = Variable(dataX).cuda()
            labelY = Variable(labelY).cuda()
            optimizer.zero_grad()
            output = model(inputX)
            loss = criterion(output, labelY)
            loss.backward()
            optimizer.step()
            # if ii % 10 == 0:
            #     print('Loss: {:.3f}'.format(loss.item()))

        model.eval()
        acc = 0
        with torch.no_grad():
            for ii, (dataX, labelY) in enumerate(test_loader):
                test_X = Variable(dataX).cuda()
                test_out = model(test_X)
                test_out = torch.sigmoid(test_out).cpu().detach().numpy()
                test_out = np.round(test_out)
                acc += accuracy_score(test_out.reshape(-1, num_classes), labelY.numpy().astype(int).reshape(-1, num_classes), normalize=False)
        acc = acc / len(test_data)
        temp_str = str(epoch) + '\t' + str(acc) + '\n'
        print('Epoch: ', epoch, 'Acc:', acc)
        filename = 'orig.txt'
        fp = open(filename, 'a')
        fp.write(temp_str)
        fp.close()

if __name__ == '__main__':
    args = parse_args()

    print('====== loading data ======')
    doc_X, doc_Y = prepare_data()
    per_client = len(doc_X) // args.nworker
    splitX = []
    splitY = []

    totalX, totalY = clean_data(doc_X, doc_Y)
    totalX, totalY = convert_words(totalX, totalY)
    totalY_train = copy.deepcopy(totalY)
    temp_Y, Yindices = np.unique(totalY, axis=0, return_inverse=True)
    totalY = Yindices
    N_class = np.max(totalY) + 1
    for i in range(args.nworker-1):
        splitX.append(totalX[i*per_client : (i+1)*per_client])
        splitY.append(totalY[i*per_client : (i+1)*per_client])
    splitX.append(totalX[(args.nworker-1)*per_client:])
    splitY.append(totalY[(args.nworker-1)*per_client:])


    if args.type == 'filter':
        print('====== calculating global table ======')
        global chi_table
        chi_table = construct_global_table(totalX, totalY)
        chi_table = get_global_E(chi_table, N_class, totalX.shape[0])

        print('====== sampling from stable distribution ======')
        global gaussian_table
        gaussian_table = server_gaussian(chi_table, N_class, args.samples)
        upload_table_sample = dict()
        upload_table = dict()

        print('====== preparing ======')
        for key in tqdm(gaussian_table.keys()):
            upload_table_sample[key] = np.zeros((args.samples, 1))
            upload_table[key] = np.zeros((2, N_class))
        global temp_sample
        temp_sample = upload_table_sample
        global temp_table
        temp_table = upload_table
        client_result = []
        pool2 = Pool(16)
        # FIXME: need too large memory
        print('====== client side ======')
        client_result = list(pool2.map(client_calculate, [(splitX[i], splitY[i], args.nworker, args.samples, N_class) for i in range(args.nworker)]))
        # for i in tqdm(range(args.nworker)):
        #     client_result.append(client_calculate((copy.deepcopy(upload_table_sample), copy.deepcopy(upload_table), splitX[i], splitY[i], gaussian_table, chi_table, args.nworker, args.samples)))

        print('====== server side ======')
        server_result = server_agg(args.samples, client_result, dropout=20)
        sorted_server_result = dict(sorted(server_result.items(), key=lambda item: item[1], reverse=True))

        server_result = np.fromiter(server_result.values(), dtype=float)

        print('====== baseline ======')
        baseline_result = baseline(totalX, totalY, chi_table, upload_table)
        sorted_baseline_result = dict(sorted(baseline_result.items(), key=lambda item: item[1], reverse=True))

        baseline_result = np.fromiter(baseline_result.values(), dtype=float)

        print('====== difference ======')
        avg_diff = np.mean(np.abs(baseline_result - server_result))
        max_diff = np.max(np.abs(baseline_result - server_result))
        print('average diff: ', avg_diff)
        print('max diff: ', max_diff)

        print('====== writing ======')
        with open('sorted_server_result.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in sorted_server_result.items():
                writer.writerow([key, value])

        with open('sorted_baseline_result.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in sorted_baseline_result.items():
                writer.writerow([key, value])
        
        print('====== filtering ======')
        fed_terms = []
        baseline_terms = []
        index = 0
        for key, _ in sorted_server_result.items():
            index += 1
            fed_terms.append(key)
            if index > 40000:
                break
        index = 0
        for key, _ in sorted_baseline_result.items():
            index += 1
            baseline_terms.append(key)
            if index > 40000:
                break

        assert len(fed_terms) == len(baseline_terms)
        with open('fed_terms.txt', 'wb') as fp:
            pickle.dump(fed_terms, fp)
        with open('baseline_terms.txt', 'wb') as fp:
            pickle.dump(baseline_terms, fp)

        same_term_num = len(set(fed_terms) & set(baseline_terms))
        print('same terms: ', same_term_num)
        print('different terms: ', len(fed_terms) - same_term_num)


    elif args.type == 'train':
        totalY = totalY_train
        print('====== loading selected features ======')
        with open('fed_terms.txt', 'rb') as fp:
            fed_terms = pickle.load(fp)
        with open('baseline_terms.txt', 'rb') as fp:
            baseline_terms = pickle.load(fp)

        print('====== selecting features ======')
        fed_data = select_features(totalX, fed_terms)
        # with open('fed_data.txt', 'wb') as fp:
        #     pickle.dump(fed_data, fp)
        # baseline_data = select_features(totalX, baseline_terms)
        # with open('baseline_data.txt', 'wb') as fp:
        #     pickle.dump(baseline_data, fp)
        
        # with open('fed_data.txt', 'rb') as fp:
        #     fed_data = pickle.load(fp)
        # with open('baseline_data.txt', 'rb') as fp:
        #     baseline_data = pickle.load(fp)
        torch.cuda.set_device(args.device)
        print('====== training models ======')
        print('====== fed result ======')
        train_LR(fed_data, copy.deepcopy(totalY), mode='fed')
        # print('====== baseline result ======')
        # train_LR(baseline_data, copy.deepcopy(totalY), mode='baseline')

    elif args.type == 'orig':
        totalY = totalY_train
        print(totalX.shape, totalY.shape)
        torch.cuda.set_device(args.device)
        print('====== selecting features ======')
        feature_num = np.max(totalX)
        print('feature nums: ', feature_num)
        orig_data = select_features_orig(totalX)
        print('====== training models ======')
        print('====== orig result ======')
        train_LR_orig(orig_data, copy.deepcopy(totalY))
