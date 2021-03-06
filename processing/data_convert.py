import sys
import html
import re
import json
import random
import linecache

SEP_A = '^A'


def processing_sent(str_):
    str_ = str_.strip()
    str_ = html.unescape(str_)
    str_ = re.sub(r'\s+', ' ', str_)
    new_str = re.sub(r'&nbsp+', ' ', str_)
    return new_str.strip()


def processing(in_path, out_path):
    fw = open(out_path, 'w', encoding='utf-8')
    with open(in_path, 'r', encoding='utf-8', errors='ignore') as fin:
        for line in fin:
            try:
                res = []
                items = line.strip().split(SEP_A)
                if len(items) < 3:
                    continue
                res.append(items[0].strip())
                for i in range(len(items)):
                    if i != 0:
                        z = json.loads(items[i])['QuestionContentTextLatexAttr']
                        res.append(z.strip())
                if len(res) > 1:
                    cline = SEP_A.join(res)
                    fw.write(cline)
                    fw.write('\n')
            except Exception as e:
                print(e)
    fw.close()
    print('Done')


def get_rand_line_from_file(path, start_id=1, range_=300):
    fin = open(path, 'r', encoding='utf8', errors='ignore')
    for _ in range(random.randint(start_id, start_id+range_)):
        try:
            next(fin)
        except StopIteration:
            fin.close()
            fin = open(path, 'r', encoding='utf8', errors='ignore')
    rand_line = next(fin)
    fin.close()
    return rand_line


def get_rand_line(path, line_no):
    return linecache.getline(path, line_no).strip()


def produce_neg_samples(in_path, out_path):
    fw = open(out_path, 'w', encoding='utf-8')
    with open(in_path, 'r', encoding='utf-8', errors='ignore') as fin:
        for i, line in enumerate(fin):
            try:
                n_iter = 1000
                if line.strip() == '':
                    continue
                query, doc = line.strip().split('^A')[0:2]
                course = json.loads(doc)['course'].strip()
                rand_line = get_rand_line(in_path, i + 1 + random.randint(1, 1000))
                rand_docs = rand_line.strip().split('^A')[1:]
                rand_course = json.loads(rand_docs[0])['course'].strip()
                print(course, rand_course)
                while n_iter > 0 and rand_course != course:
                    rand_line = get_rand_line(in_path, i + 1 + random.randint(1, 2000))
                    rand_docs = rand_line.strip().split('^A')[1:]
                    rand_course = json.loads(rand_docs[0])['course'].strip()
                    n_iter -= 1
                if n_iter <= 0:
                    continue
                rand_docs.insert(0, query)
                neg_sample = SEP_A.join(rand_docs)
                fw.write(neg_sample)
                fw.write('\n')
            except Exception as e:
                print(e)
    fw.close()
    print('Done')


def run():
    inp, oup = sys.argv[1:3]
    processing(inp, oup)
    # produce_neg_samples(inp, oup)

run()
