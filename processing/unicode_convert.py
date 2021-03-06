import sys
import html
import re
import json


SEP_A = ''


def processing_sent(str_):
    str_ = str_.strip()
    str_ = html.unescape(str_)
    new_str = re.sub(r'&nbsp+', ' ', str_)
    # new_str = re.sub(r'\s+', ' ', str_)
    return new_str


def processing(in_path, out_path):
    fw = open(out_path, 'w', encoding='utf-8')
    with open(in_path, 'r', encoding='utf-8', errors='ignore') as fin:
        for line in fin:
            try:
                cline = processing_sent(line.strip())
                items = cline.strip().split(SEP_A)
                #print(len(items))
                if len(items) < 3:
                    continue
                for i in range(len(items)):
                    if i != 0:
                        z = json.loads(items[i])
                fw.write(cline)
                fw.write('\n')
            except Exception as e:
                print('except: ', e)
    fw.close()
    print('Done')


def convert(in_path, out_path):
    fw = open(out_path, 'w', encoding='utf8')
    one_item = 0
    with open(in_path, 'r', encoding='utf8', errors='ignore') as fin:
        for line in fin:
            try:
                items = line.strip().split(SEP_A)
                if len(items) < 2:
                    one_item += 1
                    continue
                for i in range(len(items)):
                    if i != 0:
                        items[i] = items[i].encode('utf8').decode('unicode_escape')
                cline = SEP_A.join(items)
                fw.write(cline)
                fw.write('\n')
            except Exception as e:
                print(e)
    fw.close()
    print(one_item)
    print('Done')


def run():
    inp, oup = sys.argv[1:3]
    #convert(inp, oup)
    processing(inp, oup)

run()
