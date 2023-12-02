from gensim.models import KeyedVectors
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
wv = KeyedVectors.load('embs_train.kv')
from svector import svector
import time

def wvTesting():
    #print(wv['big'])
    print('similar to wonderful: ',wv.most_similar('wonderful', topn=10))
    print('similar to awful: ',wv.most_similar('awful', topn=10))
    print('similar to clown: ',wv.most_similar('clown', topn=10))
    print('similar to evil: ',wv.most_similar('evil', topn=10))
    print('similar to plant: ', wv.most_similar('plant', topn=10))

def p1q2():
    #sister - woman + man
    word1=wv['sister']-wv['woman']+wv['man']
    print('similar 1: ', wv.most_similar(word1,topn=10))
    #harder - hard + fast
    word2=wv['harder']-wv['hard']+wv['fast']
    print('similar 2: ', wv.most_similar(word2,topn=10))
    word3=wv['man']-wv['child']+wv['cat']
    print('similar 3: ', wv.most_similar(word3,topn=10))
    word4=wv['shirt']-wv['arms']+wv['legs']
    print('similar 4: ', wv.most_similar(word4,topn=10))
    word5=wv['spoon']-wv['fork']+wv['knife']
    print('similar 5: ', wv.most_similar(word5,topn=10))

#from the code given to us for HW2
def read_from(textfile):
    for line in open(textfile):
        label, words = line.strip().split("\t")
        yield (1 if label=="+" else -1, words.split())

def similarSentences():
    trainfile= 'C:/Users/15418/PycharmProjects/AI534/HW2/train.txt'
    sentences=[]
    for j, (label, words) in enumerate(read_from(trainfile), 1):
        if j==1:
            firstSentence=np.zeros((300,),dtype=float)
            count=0
            for i in range(len(words)):
                try:
                    firstSentence=firstSentence+wv[words[i]]
                    count+=1
                except:
                    pass
            firstSentence=firstSentence/count
        elif j==2:
            secondSentence=np.zeros((300,),dtype=float)
            count=0
            for i in range(len(words)):
                try:
                    secondSentence=secondSentence+wv[words[i]]
                    count+=1
                except:
                    pass
            secondSentence=secondSentence/count
        else:
            sentence=np.zeros((300,),dtype=float)
            count=0
            for i in range(len(words)):
                try:
                    sentence=sentence+wv[words[i]]
                    count+=1
                except:
                    pass
            sentence=sentence/count
            sentences.append(sentence)
    #KeyedVectors has a built in function for this
    rankingsS1=wv.cosine_similarities(firstSentence, sentences)
    newValsS1=[]
    for i in range(len(rankingsS1)):
        newValsS1.append([i,rankingsS1[i]])
    newValsS1=np.array(newValsS1)
    sortedS1=(newValsS1[newValsS1[:, 1].argsort()])
    #print(sortedS1[0])
    file = open(trainfile)
    lines = file.readlines()
    print('most similar to sentence 1: ',lines[int(sortedS1[len(sortedS1)-1][0])])
    rankingsS2=wv.cosine_similarities(secondSentence, sentences)
    newValsS2=[]
    for i in range(len(rankingsS2)):
        newValsS2.append([i,rankingsS2[i]])
    newValsS2=np.array(newValsS2)
    sortedS2=(newValsS2[newValsS2[:, 1].argsort()])
    print('most similar to sentence 2: ',lines[int(sortedS2[len(sortedS2)-1][0])])
    print('sentence 1: ',lines[0])
    print('sentence 2: ', lines[1])

def KNNsentenceEmbedding():
    trainfile= 'C:/Users/15418/PycharmProjects/AI534/HW2/train.txt'
    devFile='C:/Users/15418/PycharmProjects/AI534/HW2/dev.txt'
    sentences=[]
    devSentences=[]
    labels=[]
    devLabels=[]
    for j, (label, words) in enumerate(read_from(trainfile), 1):
        sentence = np.zeros((300,), dtype=float)
        count = 0
        for i in range(len(words)):
            try:
                sentence = sentence + wv[words[i]]
                count += 1
            except:
                pass
        sentence = sentence / count
        labels.append(label)
        sentences.append(sentence)
    for j, (label, words) in enumerate(read_from(devFile), 1):
        sentence = np.zeros((300,), dtype=float)
        count = 0
        for i in range(len(words)):
            try:
                sentence = sentence + wv[words[i]]
                count += 1
            except:
                pass
        if count>0:
            sentence = sentence / count
            devLabels.append(label)
            devSentences.append(sentence)
    for k in range(1,101,2):
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(sentences, labels)
        #correctCount = 0
        devCorrectCount = 0
        #for i in range(len(sentences)):
         #   prediction=neigh.predict([sentences[i]])
          #  if labels[i]==prediction:
           #     correctCount=correctCount+1
        for i in range(len(devSentences)):
            devPrediction=neigh.predict([devSentences[i]])
            if devLabels[i]==devPrediction:
                devCorrectCount=devCorrectCount+1
        devPercentCorrect=devCorrectCount/len(devSentences)
        #percentCorrect=correctCount/len(sentences)
        print("k=",k, " dev error rate: ", round((1-devPercentCorrect)*100,1), " (+:", round(devPercentCorrect*100,1), ")")

def KNNOneHot():
    trainfile= 'C:/Users/15418/PycharmProjects/AI534/HW2/train.txt'
    devFile='C:/Users/15418/PycharmProjects/AI534/HW2/dev.txt'
    sentences=[]
    devSentences=[]
    labels=[]
    devLabels=[]
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    #get all the words to fit to
    allWords=[]
    #for word in wv.key_to_index:
    #    allWords.append(word)
    for i, (label, words) in enumerate(read_from(trainfile), 1):
        for word in words:
            allWords.append(word)
    for i, (label, words) in enumerate(read_from(devFile), 1):
        for word in words:
            allWords.append(word)
    label_encoder = LabelEncoder()
    allWords=np.unique(allWords)
    print(len(allWords))
    label_encoder.fit(allWords)
    #vocab length to make empty vector with
    vocabLength=len(allWords)
    for j, (label, words) in enumerate(read_from(trainfile), 1):
        sentence = np.zeros((vocabLength,), dtype=float)
        #newWords=[]
        #for i in range(len(words)):
            #newWords.append(words[i])
        labelEncoded = label_encoder.transform(words)
        sentence[labelEncoded] = 1
        #sentence = sentence / count
        labels.append(label)
        sentences.append(sentence)
    print('made sentences from train')
    for j, (label, words) in enumerate(read_from(devFile), 1):
        sentence = np.zeros((vocabLength,), dtype=float)
        #count = 0
        #newWords=[]
        #for i in range(len(words)):
        #    newWords.append(words[i])
        labelEncoded = label_encoder.transform(words)
        sentence[labelEncoded] = 1
        devLabels.append(label)
        devSentences.append(sentence)
    print('made sentences from dev')
    for k in range(1,101,2):
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(sentences, labels)
        #correctCount = 0
        devCorrectCount = 0
        for i in range(len(devSentences)):
            devPrediction=neigh.predict([devSentences[i]])
            if devLabels[i]==devPrediction:
                devCorrectCount=devCorrectCount+1
        devPercentCorrect=devCorrectCount/len(devSentences)
        #percentCorrect=correctCount/len(sentences)
        print("k=",k, " dev error rate: ", round((1-devPercentCorrect)*100,1), " (+:", round(devPercentCorrect*100,1), ")")


def testing():
    trainfile= 'C:/Users/15418/PycharmProjects/AI534/HW2/train.txt'
    devFile='C:/Users/15418/PycharmProjects/AI534/HW2/dev.txt'
    sentences=[]
    devSentences=[]
    labels=[]
    devLabels=[]
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    #get all the words to fit to
    allWords=[]
    #add all the words
    for i, (label, words) in enumerate(read_from(trainfile), 1):
        for word in words:
            allWords.append(word)
    label_encoder = LabelEncoder()
    #print(allWords)
    label_encoder.fit(allWords)
    #encoder.fit(np.array(allWords).reshape(-1,1))
    #vocab length to make empty vector with
    vocabLength=len(allWords)
    for j, (label, words) in enumerate(read_from(trainfile), 1):
        sentence = np.zeros((vocabLength,), dtype=int)
        count = 0
        newWords=[]
        for i in range(len(words)):
            newWords.append(words[i])
        labelEncoded=label_encoder.transform(newWords)
        sentence[labelEncoded]=1

def test(devfile, model):
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        err += label * (model.dot(make_vector(words))) <= 0
    return err/i  # i is |D| now

def make_vector(words):
    v = svector()
    #add bias
    v['biasTerm']=1
    for word in words:
        v[word] += 1
    return v

def basicPerceptronEmbedding():
    epochs=10
    trainfile= 'C:/Users/15418/PycharmProjects/AI534/HW2/train.txt'
    devfile='C:/Users/15418/PycharmProjects/AI534/HW2/dev.txt'
    best_err = 1.
    model = np.zeros((300,), dtype=float)
    c = 0
    wa = np.zeros((300,), dtype=float)
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
            #empty array to put words that aren't one or two count
            sentence = np.zeros((300,), dtype=float)
            count = 0
            for j in range(len(words)):
                try:
                    sentence = sentence + wv[words[j]]
                    count += 1
                except:
                    pass
            sentence = sentence / count
            if label * (model.dot(sentence)) <= 0:
                updates += 1
                model += label * sentence
        dev_err = test2(devfile, model)
        best_err = min(best_err, dev_err)

        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))

#testing with embedding
def test2(devfile,model):
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        sentence = np.zeros((300,), dtype=float)
        count = 0
        for j in range(len(words)):
            try:
                sentence = sentence + wv[words[j]]
                count += 1
            except:
                pass
        sentence = sentence / count
        err += label * (model.dot(sentence)) <= 0
    return err / i



#wvTesting()
#p1q2()
#similarSentences()
#KNNsentenceEmbedding()
#testing()
#KNNOneHot()
basicPerceptronEmbedding()