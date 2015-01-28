// combine count and predic word embedding

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define _FILE_OFFSET_BITS 64
#define MAX_STRING_LENGTH 100
#define MAX_CODE_LENGTH 40

typedef double real;

typedef struct cooccur_rec {
    long long word1;
    long long word2;
    real val;
} CREC;

typedef struct vocab_word {
	long long cn;
	int *point;
	char *word, *code, codelen;
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
char vocab_file[MAX_STRING_LENGTH], input_file[MAX_STRING_LENGTH], *save_W_file, *save_gradsq_file;

int hs = 0, negative = 5;

/* Efficient string comparison */
int scmp( char *s1, char *s2 ) {
    while(*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return(*s1 - *s2);
}

void AddWordToVocab(char *word, long long count) {
	unsigned int length = strlen(word) + 1;
	if (length > MAX_STRING_LENGTH) length = MAX_STRING_LENGTH;
	vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
	strcpy(vocab[vocab_size].word, word);
	vocab[vocab_size].cn = count;

	if (vocab_size + 2 >= vocab_max_size) {
		vocab_max_size += 1000;
		vocab = (VWORD *)realloc(vocab, vocab_max_size * sizeof(VWORD));
	}
	vocab_size++; // NOTE that there is a dismatch betwen vocab index and cooccurance word index. according to glove/cooccure.c word id start form 1
}

void ReadVocab() {
	long long count, a;
	char word[MAX_STRING_LENGTH], format[20];
	FILE *fid;
	fid = fopen(vocab_file, "r");
	if (fid == NULL) {fprintf(stderr, "Unable to open vocab file %s.\n", vocab_file); exit(1);}
	sprintf(format, "%%%ds %%lld", MAX_STRING_LENGTH);
	while (fscanf(fid, format, word, &count) != EOF) {
		AddWordToVocab(word, count);
	}
	fclose(fid);

	for (a = 0; a < vocab_size; a++) {
		vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
		vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
	}
}

void CreateBinaryTree() {
	long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
	char code[MAX_CODE_LENGTH];
	long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
	for (a = vocab_size; a < 2 * vocab_size; a++) count[a] = 1e15;
	pos1 = vocab_size - 1;
	pos2 = vocab_size;

	for (a = 0; a < vocab_size - 1; a++) {
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min1i = pos1;
				pos1--;
			} else {
				min1i = pos2;
				pos2++;
			}
		} else {
			min1i = pos2;
			pos2++;
		}
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min2i = pos1;
				pos1--;
			} else {
				min2i = pos2;
				pos2++;
			}
		} else {
			min2i = pos2;
			pos2++;
		}

		count[vocab_size + a] = count[min1i] + count[min2i];
		parent_node[min1i] = vocab_size + a;
		parent_node[min2i] = vocab_size + a;
		binary[min2i] = 1;
	}

	for (a = 0; a < vocab_size; a++) {
		b = a;
		i = 0;
		while (1) {
			code[i] = binary[b];
			point[i] = b;
			i++;
			b = parent_node[b];
			if (b == vocab_size * 2 - 2) break;
		}
		vocab[a].codelen = i;
		vocab[a].point[0] = vocab_size - 2;
		for (b = 0; b < i; b++) {
			vocab[a].code[i - b - 1] = code[b];
			vocab[a].code[i - b] = point[b] - vocab_size;
		}
	}
	free(count);
	free(binary);
	free(parent_node);
}


void InitNet()
{
	long long a, b;
	unsigned long long next_random = 1;
	a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * vector_size * sizeof(real));
	if (syn0 == NULL) {fprintf(stderr, "Error allocating memory for syn0\n"), exit(1);}
	if (hs) {
		a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * vector_size * sizeof(real));
		if (syn1 == NULL) {fprintf(stderr, "Error allocating memory for syn1\n"), exit(1);}
		for (a = 0; a < vocab_size; a++) for (b = 0; b < vocab_size; b++)
			syn1[a * vocab_size + b] = 0;
	}
	if (negative > 0){
		a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * vector_size * sizeof(real));
		if (syn1 == NULL) {fprintf(stderr, "Error allocating memory for syn1\n"), exit(1);}
		for (a = 0; a < vocab_size; a++) for (b = 0; b < vocab_size; b++)
			syn1neg[a * vocab_size + b] = 0;
	}
	for (a = 0; a < vocab_size; a++) for (b = 0; b < vector_size; b++) {
		next_random = next_random * (unsigned long long)25214903917 + 11;
		syn0[a * vector_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / vector_size;
	}
	CreateBinaryTree();
}



int main()
{

	//Initialize the vocab with vocab_max_size
	strcpy(vocab_file, "vocab.txt");
	vocab = (VWORD *)calloc(vocab_max_size, sizeof(VWORD));
	ReadVocab();
	CreateBinaryTree();

	return 0;
}
