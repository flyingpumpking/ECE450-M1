def ngram_similarity(string1, string2, n):
    """
    Calculates the similarity between two strings using the n-gram algorithm.

    Args:
        string1: The first string.
        string2: The second string.
        n: The size of the n-grams.

    Returns:
        A float between 0 and 1, representing the similarity score.
    """
    if not string1 or not string2:
        return 0.0

    # Helper function to generate n-grams for a string
    def get_ngrams(s, n):
        ngrams = set()
        for i in range(len(s) - n + 1):
            ngrams.add(s[i:i + n])
        return ngrams

    ngrams1 = get_ngrams(string1, n)
    ngrams2 = get_ngrams(string2, n)

    if not ngrams1 or not ngrams2:  # Handles cases where strings are shorter than n
        return 0.0

    common_ngrams = ngrams1.intersection(ngrams2)

    # Jaccard Index for similarity
    similarity = len(common_ngrams) / (len(ngrams1) + len(ngrams2) - len(common_ngrams))

    return similarity


if __name__ == "__main__":
    # Example usage:
    string_a = "this is a test"
    string_b = "this is another test"
    n_value = 3

    similarity_score = ngram_similarity(string_a, string_b, n_value)
    print(f"The {n_value}-gram similarity between '{string_a}' and '{string_b}' is: {similarity_score}")

    string_c = "apple"
    string_d = "apply"
    similarity_score_2 = ngram_similarity(string_c, string_d, 2)
    print(f"The 2-gram similarity between '{string_c}' and '{string_d}' is: {similarity_score_2}")

    string_e = "completelydifferent"
    string_f = "nothingalike"
    similarity_score_3 = ngram_similarity(string_e, string_f, 3)
    print(f"The 3-gram similarity between '{string_e}' and '{string_f}' is: {similarity_score_3}")

    # Example with strings shorter than n
    string_g = "hi"
    string_h = "hello"
    similarity_score_4 = ngram_similarity(string_g, string_h, 3)
    print(f"The 3-gram similarity between '{string_g}' and '{string_h}' is: {similarity_score_4}")