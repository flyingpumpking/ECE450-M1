from nword_gram import ngram_similarity

A = "Use a power screwdriver to remove the refrigerator back panel/top cover, separate the plastic shell from the metal frame, and sort the screws."
B = "Clip structure for manual separation of housing from metal frame."
C = "Use a power screwdriver to remove backplate/top cover screws in bulk."
combined = A + " " + B

if __name__ == "__main__":
    for i in range(1, 11):
        ngram_sim_of_split = ngram_similarity(B, C, i)
        ngram_sim_of_combined = ngram_similarity(A, combined, i)

        print(f"n-gram similarity of split: {ngram_sim_of_split:.4f}")
        print(f"n-gram similarity of combined: {ngram_sim_of_combined:.4f}")

        if i != 10:
            print("-" * 50)
