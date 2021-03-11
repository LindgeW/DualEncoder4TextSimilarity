import jieba.analyse


# Jaccard [0, 1]
def jaccard_dist(s1, s2):
    s1_ = set(s1.lower())
    s2_ = set(s2.lower())
    inter = s1_ & s2_
    union = s1_ | s2_
    jac = len(inter) / len(union)
    return jac


def jaccard_dist_len_penalty(s1, s2, alpha=0.02):
    s1_ = set(s1.strip().lower())
    s2_ = set(s2.strip().lower())
    inter = s1_ & s2_
    union = s1_ | s2_
    len_diff = abs(len(s1.strip()) - len(s2.strip()))  # 增加文本长度惩罚
    jac = len(inter) / (len(union) + alpha * len_diff)
    return jac


# Jaro [0, 1]
def jaro_dist(s, t):
    s_len = len(s)
    t_len = len(t)

    if s_len == 0 and t_len == 0:
        return 1

    match_distance = (max(s_len, t_len) // 2) - 1

    s_matches = [False] * s_len
    t_matches = [False] * t_len

    matches = 0
    transpositions = 0

    for i in range(s_len):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, t_len)

        for j in range(start, end):
            if t_matches[j]:
                continue
            if s[i] != t[j]:
                continue
            s_matches[i] = True
            t_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0

    k = 0
    for i in range(s_len):
        if not s_matches[i]:
            continue
        while not t_matches[k]:
            k += 1
        if s[i] != t[k]:
            transpositions += 1
        k += 1

    return ((matches / s_len) + (matches / t_len) +
            ((matches - transpositions / 2) / matches)) / 3


def common_prefix_len(s1, s2):
    m = min(len(s1), len(s2))
    pl = 0
    for i in range(m):
        if s1[i] == s2[i]:
            pl += 1
        else:
            break
    return pl


# 前缀相同的字符串分数更高
def jaro_winkler_dist(s1, s2):
    scale_factor = 0.1    # (0, 0.25)
    jaro_score = jaro_dist(s1, s2)
    cpl = min(4, common_prefix_len(s1, s2))
    match_score = ((jaro_score + (scale_factor * cpl * (1 - jaro_score))) * 100) / 100
    return match_score


# 如何是词的相似度，基本单位是字符；如果是段落的相似度，基本单位是词
def edit_dist(s1, s2):
    m = len(s1)
    n = len(s2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
   
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j 
            elif j == 0:
                dp[i][j] = i
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],  dp[i - 1][j], dp[i - 1][j - 1])  
    return dp[m][n]


# Levenshtein
def lest_dist(s1, s2):
    n = len(s1)
    m = len(s2)
    if n > m:
        s1, s2 = s2, s1
        n, m = m, n

    d = [0 for _ in range(1+n)]  # cost
    p = [i for i in range(1+n)]  # previous cost
    for i in range(1, 1+m):
        s2_i = s2[i-1]
        d[0] = i
        for j in range(1, 1+n):
            cost = 0 if s1[j-1] == s2_i else 1
            d[j] = min(min(d[j - 1] + 1, p[j] + 1), p[j - 1] + cost)
        p, d = d, p
    return p[n]


if __name__ == "__main__":
    s1 = "NLP是一个不错的研究方向"
    s2 = "NLP是一个非常有意思的研究热点"
    dist = jaccard_dist(s1, s2)
    print(dist)
    # dist = edit_dist(s1, s2)
    # print(lest_dist(s1, s2))
    print(jaro_dist(s1, s2))
    print(jaro_winkler_dist(s1, s2))
