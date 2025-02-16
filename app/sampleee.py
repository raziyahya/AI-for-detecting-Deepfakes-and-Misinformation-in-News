import nltk
import numpy as np
import requests
from nltk import word_tokenize
from nltk.corpus import stopwords


def checknews(news):


    res = stop(news)
    key = ''
    i = 1
    resultset = []
    resultset1 = []

    for r in res:
        key = key + " " + r
    keyval = key

    print('http://www.google.co.in/search?rlz=1C1CHBF_enIN790IN790&biw=935&bih=657&ei=CDVcXLOCJJeRwgPorKjwDA&q=' + keyval)

    res1 = requests.get(
        'http://www.google.co.in/search?rlz=1C1CHBF_enIN790IN790&biw=935&bih=657&ei=CDVcXLOCJJeRwgPorKjwDA&q=' + keyval)
    # print(res1.text)
    resn = []

    # print(res1.text)
    import re
    clean = re.compile('<.*?>')

    # print(res1.text)
    #   class="ZINbbc xpd O9g5cc uUPGi"
    # ll = res1.text.split('<div class="ZINbbc xpd O9g5cc uUPGi"')
    ll = res1.text.split('<div class="BNeawe vvjwJb AP7Wnd">')
    print(ll)
    # print(ll)
    #
    print(len(ll), "lennnnnnnnnnnnnnnnnnnnnnnnn")
    count=len(ll)
    if count>5:
        count=5
    for i in range(1, count):
        ll1 = ll[i]

        lll=ll1.split('<div class="kCrYT">')
        print(len(lll),"+++---++++---+++---+++")
        if len(lll)>1:

            newsl=lll[1]

            text = re.sub(clean, "", newsl)
            print("==============================================================")
            print(text)

            print("==============================================================")

            resn.append(text)
    # print(resn, " urlsssssssssssssssssss")


    sim = []
    for n in resn:
        # print(n)

        dictl = process(n)

        print("dictl",n,dictl)

        dict2 = process(news)
        print("dict2",news,dict2)

        sim.append(getsimilarity(dict2, dictl))
        print("=======================================")
    #
    print("similarity between Bug#599831 and Bug#800279 is ", sim)
    sum = 0.0
    cou = 0
    print("sim",sim)
    for s in sim:
        if float(s) > 0.55:
            cou = cou + 1
        sum = sum + float(s)

    sum = sum / len(sim)
    conn = cou / len(sim)
    print(cou / len(sim))
    print(sum)
    thr = ""
    if sum >= 0.45:
        thr = "Real"
    else:
        thr = "AI GENERATED"
    # cmd.execute("insert into news values(null,'" + str(uid) + "','" + heading + "','" + news + "',curdate(),'"+thr+"')")
    # con.commit()
    # print(thr)
    print (thr)
    return thr

def getsimilarity(dictl, dict2) :
    all_words_list= []
    for  key in dictl:
        all_words_list.append(key)
    for key in dict2:
           all_words_list.append(key)
    all_words_list_size = len(all_words_list)

    v1 = np.zeros(all_words_list_size, dtype=np.int)
    v2 = np.zeros(all_words_list_size, dtype=np.int)
    i = 0
    for (key) in all_words_list:
        v1[i] = dictl.get(key, 0)
        v2[i] = dict2.get(key, 0)
        i = i+1
    return cos_sim(v1, v2)



def stop(text):
    with open(r'C:\Users\Asus\Desktop\DeepFake_5_1\Deepfake\example_txt.txt', 'r') as file:
        content = file.read()  # Reads the entire file

    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import numpy as np
    import nltk

    def process(file):
        raw = open(file).read()
        tokens = word_tokenize(raw)
        words = [w.lower() for w in tokens]
        porter = nltk.PorterStemmer()
        stemmed_tokens = [porter.stem(t) for t in words]
        # Removing stop words
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [w for w in stemmed_tokens if not w in stop_words]
        # count words
        count = nltk.defaultdict(int)
        for word in filtered_tokens:
            count[word] += 1
        return count;

    def cos_sim(a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def getsimilarity(dictl, dict2):
        all_words_list = []
        for key in dictl:
            all_words_list.append(key)
        for key in dict2:
            all_words_list.append(key)
        all_words_list_size = len(all_words_list)

        v1 = np.zeros(all_words_list_size, dtype=np.int)
        v2 = np.zeros(all_words_list_size, dtype=np.int)
        i = 0
        for (key) in all_words_list:
            v1[i] = dictl.get(key, 0)
            v2[i] = dict2.get(key, 0)
            i = i + 1
        return cos_sim(v1, v2)
    try:
        print(content.lower().split("\n"))
        example_sent = content.lower().split("\n")[0][:400]
        print(example_sent)
    except:
        example_sent = content.lower()
    example_sent=str(example_sent).replace('-',' ')
    example_sent = str(example_sent).replace('_', ' ')
    stop_words= set(stopwords.words('english'))
    word_tokens= word_tokenize(example_sent)

    filtered_sentence= [w for w in word_tokens if not w in stop_words]

    filtered_sentence=[]

    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)


    return filtered_sentence
def process(file):
    raw =file
    tokens = word_tokenize(raw)
    words = [w. lower() for w in tokens]
    porter = nltk.PorterStemmer()
    stemmed_tokens = [porter. stem(t) for t in words]
    # Removing stop words
    stop_words = set(stopwords.words( 'english' ) )
    filtered_tokens = [w for w in stemmed_tokens if not w in stop_words]
    # count words
    count = nltk.defaultdict(int)
    for word in filtered_tokens:
        count [word] += 1
    return count;
def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a  * norm_b)




# print(checknews("sachin tendulkar  dead in an accident"))




