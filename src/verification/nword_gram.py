# get_ngrams 函数，用于生成基于词的 n-gram
def get_ngrams(text, n_words):
    """
    Generates word-based n-grams for a string.
    Words are separated by spaces. An n-gram of 3 words will have 2 spaces.

    Args:
        text: The input string.
        n_words: The number of words in each n-gram.

    Returns:
        A set of n-gram strings.
    """
    words = text.split()  # 按空格分割字符串成词语列表

    # 如果词语数量少于 n_words，则无法形成任何 n-gram
    if len(words) < n_words:
        return set()

    ngrams_set = set()
    # 遍历词语列表以提取 n-gram
    for i in range(len(words) - n_words + 1):
        # 选取 n_words 个连续的词
        ngram_tuple = words[i: i + n_words]
        # 将词语用单个空格连接起来形成 n-gram 字符串
        ngrams_set.add(" ".join(ngram_tuple))
    return ngrams_set


# ngram_similarity 函数，使用基于词的 get_ngrams
def ngram_similarity(string1, string2, n):
    """
    Calculates the similarity between two strings using word-based n-grams.

    Args:
        string1: The first string.
        string2: The second string.
        n: The number of words in each n-gram (e.g., 3 for 3-grams).

    Returns:
        A float between 0 and 1, representing the similarity score.
    """
    # 处理空字符串的边界情况
    if not string1 and not string2:  # 两者都为空
        return 1.0
    if not string1 or not string2:  # 其中一个为空
        return 0.0

    # 使用基于词的 get_ngrams 函数生成 n-gram 集合
    ngrams1 = get_ngrams(string1, n)
    ngrams2 = get_ngrams(string2, n)

    # 如果两个字符串都太短，无法生成任何指定长度的 n-gram
    if not ngrams1 and not ngrams2:
        # 如果原始字符串完全相同，则相似度为1，否则为0
        return 1.0 if string1 == string2 else 0.0

    # 如果只有一个字符串能生成 n-gram (另一个太短)
    if not ngrams1 or not ngrams2:
        return 0.0

    # 计算共同的 n-gram
    common_ngrams = ngrams1.intersection(ngrams2)

    # 使用 Jaccard 相似系数计算相似度: |A ∩ B| / |A ∪ B|
    # |A ∪ B| = |A| + |B| - |A ∩ B|
    # 因为在此之前的检查确保了 ngrams1 和 ngrams2 至少有一个非空，
    # 并且如果一个为空另一个非空也已处理，所以此处分母不会为0，
    # 除非两个集合都为空（这种情况已被上面的 if 条件覆盖）。
    # 若执行到这里，ngrams1 和 ngrams2 都是非空集合。
    denominator = len(ngrams1) + len(ngrams2) - len(common_ngrams)

    if denominator == 0:  # Should only happen if ngrams1 and ngrams2 were empty and identical, handled above.
        # Or if both are non-empty and identical.
        return 1.0 if len(common_ngrams) > 0 else 0.0

    similarity = len(common_ngrams) / denominator

    return similarity


if __name__ == '__main__':
    # 示例用法:
    string_a = "this is a test string for ngrams"
    string_b = "this is another test phrase for ngrams"
    n_value = 3  # 表示 3-word grams

    # 测试 get_ngrams
    print(f"N-grams for '{string_a}' with n={n_value}: {get_ngrams(string_a, n_value)}")
    print(f"N-grams for '{string_b}' with n={n_value}: {get_ngrams(string_b, n_value)}")
    print("---")

    similarity_score = ngram_similarity(string_a, string_b, n_value)
    print(f"The {n_value}-word gram similarity between '{string_a}' and '{string_b}' is: {similarity_score}")
    print("---")

    string_c = "apple banana cherry"
    string_d = "banana cherry date"
    n_value_2 = 2  # 2-word grams
    similarity_score_2 = ngram_similarity(string_c, string_d, n_value_2)
    print(f"The {n_value_2}-word gram similarity between '{string_c}' and '{string_d}' is: {similarity_score_2}")
    print(f"N-grams for '{string_c}' with n={n_value_2}: {get_ngrams(string_c, n_value_2)}")
    print(f"N-grams for '{string_d}' with n={n_value_2}: {get_ngrams(string_d, n_value_2)}")
    print("---")

    string_e = "very short"
    string_f = "also short"
    n_value_3 = 3
    similarity_score_3 = ngram_similarity(string_e, string_f, n_value_3)
    print(
        f"The {n_value_3}-word gram similarity between '{string_e}' and '{string_f}' is: {similarity_score_3} (expected 0.0 as n-grams are empty and strings differ)")
    print(f"N-grams for '{string_e}' with n={n_value_3}: {get_ngrams(string_e, n_value_3)}")
    print(f"N-grams for '{string_f}' with n={n_value_3}: {get_ngrams(string_f, n_value_3)}")
    print("---")

    string_g = "identical"
    string_h = "identical"
    n_value_4 = 2
    similarity_score_4 = ngram_similarity(string_g, string_h, n_value_4)
    print(
        f"The {n_value_4}-word gram similarity between '{string_g}' and '{string_h}' is: {similarity_score_4} (expected 1.0 as n-grams are empty but strings are identical)")
    print(f"N-grams for '{string_g}' with n={n_value_4}: {get_ngrams(string_g, n_value_4)}")
    print("---")

    string_i = "a b c"
    string_j = "a b c"
    n_value_5 = 3
    similarity_score_5 = ngram_similarity(string_i, string_j, n_value_5)
    print(
        f"The {n_value_5}-word gram similarity between '{string_i}' and '{string_j}' is: {similarity_score_5} (expected 1.0)")
    print(f"N-grams for '{string_i}' with n={n_value_5}: {get_ngrams(string_i, n_value_5)}")
