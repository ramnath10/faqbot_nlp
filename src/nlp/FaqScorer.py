import nltk
import csv


def read_faq_from_file(file_path):
    file = open(file_path, 'r')
    return file.readlines()


def remove_punctuations_and_lemmatize(words):
    wnl = nltk.WordNetLemmatizer()
    return [wnl.lemmatize(word) for word in words if word.isalpha()]


def create_vocabulary(text_array):
    text = ''
    for txt in text_array:
        text += txt
    all_words = nltk.word_tokenize(text)
    return remove_punctuations_and_lemmatize(set(all_words))


def generate_score_for_faqs(faq_list, vocabulary):
    score_dict = {}
    for faq in faq_list:
        raw_faq_words = nltk.word_tokenize(faq)
        faq_words = remove_punctuations_and_lemmatize(raw_faq_words)
        score_dict[faq] = ['1' if word in faq_words else '0' for word in vocabulary]
    return score_dict


def write_score_to_csv(score_dict, filepath):
    with open(filepath, 'w') as f:
        w = csv.DictWriter(f, score_dict.keys())
        w.writeheader()
        w.writerow(score_dict)


def main():
    faq_list = read_faq_from_file('E:/PyCharm Workspace/faqbot_nlp/resources/faq.txt')
    print('File contents', faq_list)
    vocabulary = create_vocabulary(faq_list)
    print('Our Vocabulary consist of these words', vocabulary)
    print('Vocabulary size', len(vocabulary))
    score_dict = generate_score_for_faqs(faq_list, vocabulary)
    print(score_dict)
    write_score_to_csv(score_dict, 'E:/PyCharm Workspace/faqbot_nlp/resources/faq_scores.csv')


if __name__ == '__main__':
    main()

