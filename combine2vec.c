// combine count and predic word embedding

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define _FILE_OFFSET_BITS 64
#define MAX_STRING_LENGTH 1000

typedef double real;

typedef struct cooccur_rec {
    long long word1;
    long long word2;
    real val;
} CREC;

typedef struct vocab_word {
	long long cn;
	int *point;
	char *word, *code, codelent;
} VWORD;

int verbose = 2; // 0, 1, or 2
int num_threads = 8; // pthreads
int num_iter = 25; // Number of full passes through cooccurrence matrix
int save_gradsq = 0; // By default don't save squared gradient values
int use_binary = 1; // 0: save as text files; 1: save as binary; 2: both. For binary, save both word and context word vectors.
int model = 2; // For text file output only. 0: concatenate word and context vectors (and biases) i.e. save everything; 1: Just save word vectors (no bias); 2: Save (word + context word) vectors (no biases)
real eta = 0.05; // Initial learning rate
real alpha = 0.75, x_max = 100.0; // Weighting function parameters, not extremely sensitive to corpus, though may need adjustment for very small or very large corpora
// real *W, *gradsq, *cost;
// Two strategy for word representation. input & output (current & context) is the same or different.
// This is the first strategy, input & output are the same.
real *syn0, *syn1, *syn1neg, *gradsq, *expTable; //syn0 input word embeding (the i in glove Xij)
VWORD *vocab;
int vector_size = 50; // Word vector size
long long num_lines = 0, *lines_per_thread, vocab_size = 0, vocab_max_size = 2500;
char *vocab_file, *input_file, *save_W_file, *save_gradsq_file;

/* Efficient string comparison */
int scmp( char *s1, char *s2 ) {
    while(*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return(*s1 - *s2);
}

void AddWordToVocab(char *word, long long count) {
	unsigned int length = strlen(word) + 1;
	if (length > MAX_STRING_LENGTH) length = MAX_STRING_LENGTH;
	vocab_size++; // according to glove/cooccure.c word id start form 1
	vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
	strcpy(vocab[vocab_size].word, word);
	vocab[vocab_size].cn = count;

	if (vocab_size + 2 >= vocab_max_size) {
		vocab_max_size += 1000;
		vocab = (VWORD *)realloc(vocab, vocab_max_size * sizeof(VWORD));
	}
}

void ReadVocab() {
	long long count;
	char word[MAX_STRING_LENGTH], format[20];
	FILE *fid;
	fid = fopen(vocab_file, "r");
	if (fid == NULL) {fprint(stderr, "Unable to open vocab file %s.\n", vocab_file); exit(1);}
	sprintf("%%%ds %%lld", MAX_STRING_LENGTH);
	while (fscanf(fid, format, word, &count) != EOF) {
		AddWordToVocab(word, count);
	}
	flocse(fid);
}



/*
void InitNet()
{
	long long a, b;
	unsigned long long next_random = 1;
	a = posix_mealign((void **)&syn0, 128, (long long)vocab_size * vector_size * sizeof(real));

}
*/

int main()
{

	//Initialize the vocab with vocab_max_size
	vocab = (VWORD *)calloc(vocab_max_size, sizeof(VWORD));

}
