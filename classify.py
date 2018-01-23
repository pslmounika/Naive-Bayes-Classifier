#!/usr/bin/env python
from collections import defaultdict
from csv import DictReader, DictWriter

import nltk
import codecs
import string
import sys
from nltk.corpus import wordnet as wn
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from string import punctuation
# from nltk import pos_tag
from nltk.util import ngrams

from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import brown


kTOKENIZER = TreebankWordTokenizer()


lancaster_stemmer = LancasterStemmer()
def morphy_stem(word):
    """

    """
    punctuationLst = {'~', ':', "'", '+', '[', '\\', '@', '^', '{', '%', '(', '-', '"', '*', '|', ',', '&', '<', '`',
                      '}', '.', '_', '=', ']', '!', '>', ';', '?', '#', '$', ')', '/'}
    stem = wn.morphy(word.lower())
    #print("word-->stem-->lemma",word,stem,lancaster_stemmer.stem(word.lower()))

    if stem:
        # print("+++stem12", stem)
        return stem.lower()
    else:
        return word.lower()

"""
def num_syllables(self, word):
    word = word.lower()
    if word in self._pronunciations.keys():
        lst = self._pronunciations[word]
        syllable = []
        for proInst in lst:
            i = 0
            for ch in proInst:
                if (any(j.isdigit() for j in ch)):
                    i = i + 1
            syllable.append(i)
        return min(syllable)

    else:
        return 1
        """


class FeatureExtractor:
    def __init__(self):
        """
        You may want to add code here
        """
        self.punctuationLst = {'~', ':', "'", '+', '[', '\\', '@', '^', '{', '%', '(', '-', '"', '*', '|', ',', '&',
                               '<',
                               '`', '}', '.', '_', '=', ']', '!', '>', ';', '?', '#', '$', ')', '/'}
        self.stopsWords = set(stopwords.words("english"))
        BrownRWords=brown.words(categories="romance")
        self.BrownRWords_normalized=nltk.FreqDist(word.lower() for word in BrownRWords)
        BrownAWords = brown.words(categories="adventure")
        self.BrownAWords_normalized = nltk.FreqDist(word.lower() for word in BrownAWords)
        #self.suffixList=["'s","'d"]
        self._pronunciations = nltk.corpus.cmudict.dict()

        None

    def features(self, text):

        d = defaultdict(int)
        d["LongString"] = 0
        d["UpperCase"]=0
        d["Eachword"] =0
        d["AvgLen"] =0
        d["FirstWord"] =0
        d["nWords"] = 0
        d["Adventure"] = 0
        d["Romantic"] = 0

        FirstWordSet=False

        # d["Syllables"]=0
        # d["Digit"]=0
        # d["UpperCase"]=0
        # d["RomanticWords"]=0
        # romanticWords=brown.words(categories='romance')
        # d["UpperCaseCharacters"]=0
        # punctuationLst = {'~', ':', "'", '+', '[', '\\', '@', '^', '{', '%', '(', '-', '"', '*', '|', ',', '&', '<',
        # '`', '}', '.', '_', '=', ']', '!', '>', ';', '?', '#', '$', ')', '/'}
        # print("++",punctuationLst)
        tag = ""
        # stopsWords = set(stopwords.words("english"))
        tempList = []
        adWords=0
        roWords=0

        # if len(text)>=40:
        d["LongString"] = len(text)

        for ii in kTOKENIZER.tokenize(text):
            # or ii not in punctuationLst

            """
            if ii in self._pronunciations.keys():
                d["TotalSyllable"]+=len( [ph for ph in self._pronunciations[ii] if ph[0].strip(string.letters)] )
            else:
                d["TotalSyllable"] +=1"""

            d["Eachword"] = len(ii)
            d["AvgLen"] = len(ii) / len(text)
            nWords=0



            if ii not in self.stopsWords:
                d[morphy_stem(ii)] += 1
                nWords+=1
                if not FirstWordSet and ii not in self.punctuationLst and ii not in self.stopsWords:
                    d["FirstWord"]=ii.lower()
                    FirstWordSet=True
                #if ii in self.suffixList:
                #    d["suffixes"] += 1
                if ii in self.BrownRWords_normalized.keys() and ii not in self.BrownAWords_normalized.keys():
                    roWords += 1
                elif ii not in self.BrownRWords_normalized.keys() and ii in self.BrownAWords_normalized.keys():
                    adWords += 1
                elif ii in self.BrownRWords_normalized.keys() and ii in self.BrownAWords_normalized.keys():
                    if self.BrownRWords_normalized[ii] >= self.BrownAWords_normalized[ii]:
                        roWords += 1
                    else:
                        adWords += 1

                # if ii.isdigit():
                # d["Digit"]+=1
                if ii.isupper() and len(ii)>1 and ii not in self.punctuationLst:
                 d["UpperCase"] += 1

                # if ii.isupper():
                # 	d["UpperCaseCharacters"]+=1
                # if ii not in punctuationLst:
            tempList.append(ii.lower())
            d["nWords"]=nWords
            if adWords > roWords:
                d["Adventure"]=adWords
            else:
                d["Romantic"]=roWords
            for n in range(2, 7):
                for ngram in ngrams(tempList, n):
                    d[ngram] += 1
                    # tags = [t for w, t in nltk.pos_tag(ngram)]
                    # tag = "-".join(x for x in tags if x not in self.punctuationLst)
                    # d[tag]+=1
                    # print("ngram-->tag",ngram,tag)

        # tags = [t for w, t in nltk.pos_tag(kTOKENIZER.tokenize(text))]
        # tag="-".join(x for x in tags if x not in punctuationLst)
        # d[tag]+=1

        # d["POSTag"]=tag
        # print("text,syllables",text,d["Syllables"])
        return d


reader = codecs.getreader('utf8')
writer = codecs.getwriter('utf8')


def prepfile(fh, code):
    if type(fh) is str:
        fh = open(fh, code)
    ret = gzip.open(fh.name, code if code.endswith("t") else code + "t") if fh.name.endswith(".gz") else fh
    if sys.version_info[0] == 2:
        if code.startswith('r'):
            ret = reader(fh)
        elif code.startswith('w'):
            ret = writer(fh)
        else:
            sys.stderr.write("I didn't understand code " + code + "\n")
            sys.exit(1)
    return ret


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--trainfile", "-i", nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                        help="input train file")
    parser.add_argument("--testfile", "-t", nargs='?', type=argparse.FileType('r'), default=None,
                        help="input test file")
    parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout,
                        help="output file")
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='subsample this fraction of total')
    args = parser.parse_args()
    trainfile = prepfile(args.trainfile, 'r')
    if args.testfile is not None:
        testfile = prepfile(args.testfile, 'r')
    else:
        testfile = None
    outfile = prepfile(args.outfile, 'w')

    # Create feature extractor (you may want to modify this)
    fe = FeatureExtractor()

    # Read in training data
    train = DictReader(trainfile, delimiter='\t')

    # Split off dev section
    dev_train = []
    dev_test = []
    full_train = []

    for ii in train:
        if args.subsample < 1.0 and int(ii['id']) % 100 > 100 * args.subsample:
            continue
        feat = fe.features(ii['text'])
        if int(ii['id']) % 5 == 0:
            dev_test.append((feat, ii['cat']))
        else:
            dev_train.append((feat, ii['cat']))
        full_train.append((feat, ii['cat']))

    # Train a classifier
    sys.stderr.write("Training classifier ...\n")
    classifier = nltk.classify.NaiveBayesClassifier.train(dev_train)
    # print "%.3f" % nltk.classify.accuracy(classifier, dev_test)
    classifier.show_most_informative_features(200)
    # classifier.prob_classify(featurize(name))

    right = 0
    total = len(dev_test)
    for ii in dev_test:
        prediction = classifier.classify(ii[0])
        if prediction == ii[1]:
            right += 1
    sys.stderr.write("Accuracy on dev: %f\n" % (float(right) / float(total)))

    if testfile is None:
        sys.stderr.write("No test file passed; stopping.\n")
    else:
        # Retrain on all data
        classifier = nltk.classify.NaiveBayesClassifier.train(dev_train + dev_test)

        # Read in test section
        test = {}
        for ii in DictReader(testfile, delimiter='\t'):
            test[ii['id']] = classifier.classify(fe.features(ii['text']))

        # Write predictions
        o = DictWriter(outfile, ['id', 'pred'])
        o.writeheader()
        for ii in sorted(test):
            o.writerow({'id': ii, 'pred': test[ii]})


