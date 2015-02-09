// combine count and predic word embedding
// Two strategy for word representation. input & output (current & context) is the same or different.
// This is the first strategy, input & output are the same.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define _FILE_OFFSET_BITS 64
#define MAX_STRING_LENGTH 100
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define MAX_EXP 6
#define EXP_TABLE_SIZE 1000

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef double real;

typedef struct cooccur_rec {
    int word1;
    int word2;
    real val;
} CREC;

typedef struct vocab_word {
	long long cn;
	int *point;
	char *word, *code, codelen;
} VWORD;

typedef struct bi_word{
	char *biword;
	real count;
} BIWORD;

int verbose = 2; // 0, 1, or 2
int binary = 1;
int num_threads = 1; // pthreads
int adagrad = 0;
int num_iter = 5; // Number of full passes through cooccurrence matrix
int save_gradsq = 0; // By default don't save squared gradient values
int vector_size = 100; // Word vector size
int window = 5;
int *vocab_hash;
long long num_lines = 0, train_file_size = 0, vocab_size = 0, vocab_max_size = 2500;
long long word_count_actual = 0, train_words = 0;;
long long bigram_hash_size;
char vocab_file[MAX_STRING_LENGTH], train_file[MAX_STRING_LENGTH], output_file[MAX_STRING_LENGTH], cooccur_file[MAX_STRING_LENGTH];
real learn_rate = 0.025, starting_learn_rate, sample = 1e-3; // Initial learning rate
real alpha = 0.75, x_max = 100.0; // Weighting function parameters, not extremely sensitive to corpus, though may need adjustment for very small or very large corpora
real *syn0, *syn1, *syn1neg, *expTable; //syn0 input word embeding (the i in glove Xij)
real *syn0_gradsq, *syn1_gradsq, *syn1neg_gradsq;
real *predict_cost, *count_cost;
VWORD *vocab;
BIWORD **bigram_table;
clock_t start;

int hs = 1, negative = 0;
const int table_size = 1e8;
int *table;

/* Efficient string comparison */
int scmp( char *s1, char *s2 ) {
    while(*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return(*s1 - *s2);
}


/*************************Constract a bigram table******************************/
char *ConcatenateWord(char* word1, char* word2) {
	char *biword;
	biword = (char *)calloc(strlen(word1) + strlen(word2) + 2, sizeof(char));
	strcpy(biword, word1);
	strcat(biword, "#");
	strcat(biword, word2);
	return biword;
}

char *ConcatenateWordIdx(long long word1_idx, long long word2_idx) {
	char *biword, *word1, *word2;
	word1 = vocab[word1_idx].word;
	word2 = vocab[word2_idx].word;
	biword = (char *)calloc(strlen(word1) + strlen(word2) + 2, sizeof(char));
	strcpy(biword, word1);
	strcat(biword, "#");
	strcat(biword, word2);
	return biword;
}

long long GetWordHash(char *word, long long hash_size) {
	unsigned long long a, hash = 0;
	for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
	hash = hash % hash_size;
	return hash;
}

void SaveCREC(CREC *cr) {
	char *biword, *word1, *word2;
	unsigned long long hash;
	word1 = vocab[cr->word1].word; // the first word of vocab is </s>, so word index start from 1 now.
	word2 = vocab[cr->word2].word; // there is not need to minus 1.
	biword = ConcatenateWord(word1, word2);
	hash = GetWordHash(biword, bigram_hash_size);
	while (bigram_table[hash] != NULL) hash = (hash + 1) % bigram_hash_size;
	bigram_table[hash] = (BIWORD *)malloc(sizeof(BIWORD));
	bigram_table[hash]->biword = biword;
	bigram_table[hash]->count = cr->val;
}

real GetCount(char *biword) {
	unsigned long long hash = GetWordHash(biword, bigram_hash_size);
	while (1) {
		if (bigram_table[hash] == NULL) return 0.0;
		if (!strcmp(biword, bigram_table[hash]->biword)) return bigram_table[hash]->count;
		hash = (hash + 1) % bigram_hash_size;
	}
	return 0.0;
}

void ConstructBigramTable() {
	FILE *fin;
	CREC cr;
	long long a, n = 0, file_size;

	fin = fopen(cooccur_file, "rb");
	if (fin == NULL) {fprintf(stderr, "Unable to open coocurrence file %s.\n", train_file); exit(1);}
	fseek(fin, 0, SEEK_END);
	file_size = ftello(fin);
	num_lines = file_size / sizeof(CREC);

	bigram_hash_size = (long long)num_lines / 0.7;
	bigram_table = (BIWORD **)malloc(bigram_hash_size * sizeof(BIWORD *));
	for (a = 0; a < bigram_hash_size; a++) bigram_table[a] = (BIWORD *)NULL;

	fprintf(stderr, "reading bigram file %s into memory\n", cooccur_file);
	fseek(fin, 0, SEEK_SET);
	while (1) {
		n++;
		if (n % 1000000 == 0) fprintf(stderr, "%.2lf%% read into the memory\n", (n / (real)num_lines * 100));
		fread(&cr, sizeof(CREC), 1, fin);
		if (feof(fin)) break;
		SaveCREC(&cr);
	}
	fclose(fin);
}
/************************************************************************************/
// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING_LENGTH - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word, vocab_hash_size);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING_LENGTH];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

void InitUnigramTable() {
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

void AddWordToVocab(char *word, long long count) {
	unsigned int hash, length = strlen(word) + 1;
	if (length > MAX_STRING_LENGTH) length = MAX_STRING_LENGTH;
	vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
	strcpy(vocab[vocab_size].word, word);
	vocab[vocab_size].cn = count;
	train_words += count;

	if (vocab_size + 2 >= vocab_max_size) {
		vocab_max_size += 1000;
		vocab = (VWORD *)realloc(vocab, vocab_max_size * sizeof(VWORD));
	}

	hash = GetWordHash(word, vocab_hash_size);
	while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
	vocab_hash[hash] = vocab_size;
	vocab_size++; // NOTE that there is a mismatch betwen vocab index and cooccurance word index. according to glove/cooccure.c word id start form 1
}


void ReadVocab() {
	long long count, a;
	char word[MAX_STRING_LENGTH], format[20];
	FILE *fid;
	fid = fopen(vocab_file, "r");
	if (fid == NULL) {fprintf(stderr, "Unable to open vocab file %s.\n", vocab_file); exit(1);}
	sprintf(format, "%%%ds %%lld", MAX_STRING_LENGTH);

	vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;

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
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
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
  // Now assign binary code to each vocabulary word
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
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void OutputBinaryTree() {
	long long a, b;
	FILE* fout;
	fout = fopen("binary_tree.txt", "w");
	for (a = 0; a < vocab_size; a++) {
		fprintf(fout, "%s %d ", vocab[a].word, (int)vocab[a].codelen);
		for (b = 0; b < vocab[a].codelen; b++) fprintf(fout, "%d", (int)vocab[a].code[b]);
		fprintf(fout, "\n");
	}
	fclose(fout);
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
		for (a = 0; a < vocab_size; a++) for (b = 0; b < vector_size; b++)
			syn1[a * vector_size + b] = 0;
		if (adagrad) {
			a = posix_memalign((void **)&syn1_gradsq, 128, (long long)vocab_size * vector_size * sizeof(real));
			if (syn1_gradsq == NULL) {fprintf(stderr, "Error allocating memory for syn1_gradsq\n"), exit(1);}
			for (a = 0; a < vocab_size; a++) for (b = 0; b < vector_size; b++)
				syn1_gradsq[a * vector_size + b] = 1.0;
		}
	}

	a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * vector_size * sizeof(real));
	if (syn1neg == NULL) {fprintf(stderr, "Error allocating memory for syn1\n"), exit(1);}
	for (a = 0; a < vocab_size; a++) for (b = 0; b < vector_size; b++)
		syn1neg[a * vector_size + b] = 0;
	if (adagrad) {
		a = posix_memalign((void **)&syn1neg_gradsq, 128, (long long)vocab_size * vector_size *sizeof(real));
		if (syn1neg_gradsq == NULL) {fprintf(stderr, "Error allocating memory for syn1_gradsq\n"), exit(1);}
		for (a = 0; a < vocab_size; a++) for (b = 0; b < vector_size; b++)
			syn1neg_gradsq[a * vector_size + b] = 1.0;
	}

	for (a = 0; a < vocab_size; a++) for (b = 0; b < vector_size; b++) {
		next_random = next_random * (unsigned long long)25214903917 + 11;
		syn0[a * vector_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / vector_size;
	}
	if (adagrad) {
		a = posix_memalign((void **)&syn0_gradsq, 128, (long long)vocab_size * vector_size * sizeof(real));
		if (syn0_gradsq == NULL) {fprintf(stderr, "Error allocating memory for syn0_gradsq\n"), exit(1);}
		for (a = 0; a < vocab_size; a++) for(b = 0; b < vector_size; b++)
			syn0_gradsq[a * vector_size + b] = 1.0;
	}
	CreateBinaryTree();
	// OutputBinaryTree();
	// SaveParameters();
}


void *TrainModelThread(void *vid) {
	long long a, b, c, d;
	long long l1, l2, word, last_word, sentence_length = 0, sentence_position = 0;
	long long target, label, sen[MAX_SENTENCE_LENGTH + 1];
	long long local_iter = num_iter, word_count = 0, pair_count = 0, last_word_count = 0;
	long long id = (long long) vid;
	unsigned long long next_random = id;
	char *biword;
	real f, predict_grad, count_grad, f_count_grad;
	real biword_count_val;
	real temp;
	real *neu1e = (real *)calloc(vector_size, sizeof(real));
	real *neu1e_output = (real *)calloc(vector_size, sizeof(real)); // when using adagrad, neu1e is the error for output (context) word
	clock_t now;
	FILE *fin;
	fin = fopen(train_file, "rb");
	fseek(fin, train_file_size / (long long)num_threads * (long long)id, SEEK_SET);
	predict_cost[id] = 0;
	count_cost[id] = 0;

	while(1){
		if (word_count - last_word_count > 10000) {
			word_count_actual += word_count - last_word_count;
			last_word_count = word_count;
			if (verbose > 1) {
				now = clock();
				if (!adagrad) {
					fprintf(stderr, "%cLearning Rate: %f Progess: %.2f%% Words/thread/sec: %.2fk \n"
							"Predict cost/word: %.5f  Count cost/word: %.5f  Cost/word: %.5f  ", 13,
							learn_rate, word_count_actual / (real)(num_iter * train_words + 1) * 100,
							word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000),
							predict_cost[id] / pair_count, count_cost[id] / pair_count,
							(predict_cost[id] + count_cost[id]) / pair_count);
					fflush(stderr);
				} else {
					fprintf(stderr, "%cProgess: %.2f%% Words/thread/sec: %.2fk  \n"
							"Predict cost/word: %.5f  Count cost/word: %.5f  Cost/word: %.5f  ", 13,
							word_count_actual / (real)(num_iter * train_words + 1) * 100,
							word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000),
							predict_cost[id] / pair_count, count_cost[id] / pair_count,
							(predict_cost[id] + count_cost[id]) / pair_count);
					fflush(stderr);
				}
			}
			if (!adagrad) {
				learn_rate = starting_learn_rate * (1 - word_count_actual / (real)(num_iter * train_words + 1));
				if (learn_rate < starting_learn_rate * 0.0001) learn_rate = starting_learn_rate * 0.0001;
			}
		}

		if (sentence_length == 0) {
			while (1) {
				word = ReadWordIndex(fin);
				if (feof(fin)) break;
				if (word == -1) continue;
				word_count++;
				if (word == 0) break;
				// The subsampling randomly discards frequent words while keeping the ranking same
				if (sample > 0) {
					real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
					next_random = next_random * (unsigned long long)25214903917 + 11;
					if (ran < (next_random & 0xFFFF) / (real)65536) continue;
				}
				sen[sentence_length] = word;
				sentence_length++;
				if (sentence_length >= MAX_SENTENCE_LENGTH) break;
			}
			sentence_position = 0;
		}
		if (feof(fin) || (word_count >= train_words / num_threads)) {
			word_count_actual += word_count - last_word_count;
			local_iter--;
			if (local_iter == 0) break;
			word_count = 0;
			last_word_count = 0;
			sentence_length = 0;
			pair_count = 0;
			predict_cost[id] = 0;
			count_cost[id] = 0;
			fseeko(fin, (train_file_size / num_threads * id) * sizeof(CREC), SEEK_SET);
			continue;
		}

		word = sen[sentence_position]; // input word (current word)
		if (word == -1) continue;
		next_random = next_random * (unsigned long long)25214903917 + 11;
		b = next_random % window;
		for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
			c = sentence_position - window + a;
			if (c < 0) continue;
			if (c >= sentence_length) continue;
			last_word = sen[c];
			if (last_word == -1) continue;
			l1 = last_word * vector_size;
			for (c = 0; c < vector_size; c++) neu1e[c] = 0;
			for (c = 0; c < vector_size; c++) neu1e_output[c] = 0;
			if (hs) {
				// Compute preidcit error
				for (d = 0; d < vocab[word].codelen; d++) {
					f = 0;
					l2 = vocab[word].point[d] * vector_size;

					for (c = 0; c < vector_size; c++) f += syn0[c + l1] * syn1[c + l2];
					if (f <= -MAX_EXP) continue;
					else if (f >= MAX_EXP) continue;
					else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

					predict_cost[id] -= (1 - vocab[word].code[d]) * log(f) + vocab[word].code[d] * log(1 - f);

					predict_grad = (1 - vocab[word].code[d] - f) * learn_rate;
					if (!adagrad) {
						for (c = 0; c < vector_size; c++) neu1e[c] += predict_grad * syn1[c + l2];
						for (c = 0; c < vector_size; c++) syn1[c + l2] += predict_grad * syn0[c + l1];
					} else {
						for (c = 0; c < vector_size; c++) neu1e[c] += predict_grad * syn1[c + l2];
						for (c = 0; c < vector_size; c++) {
							temp = predict_grad * syn0[c + l1];
							syn1[c + l2] += temp / sqrt(syn1_gradsq[c + l2]);
							syn1_gradsq[c + l2] += temp * temp;
						}
					}
				}
			}
			if (negative > 0) {
				// Compute predict error
				for (d = 0; d < negative + 1; d++) {
					if (d == 0) {
						target = word;
						label = 1;
					} else {
						next_random = next_random * (unsigned long long)25214903917 + 11;
						target = table[(next_random >> 16) % table_size];
						if (target == 0) target = next_random % (vocab_size - 1) + 1;
						if (target == word) continue;
						label = 0;
					}

					l2 = target * vector_size;
					f = 0;
					for (c = 0; c < vector_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
					if (d == 0) count_grad = f; // elementwise multiplication is the same for count and predict
					if (f > MAX_EXP) f = 1;
					else if (f < -MAX_EXP) f = 0;
					else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

					if (f != 1 && f != 0) predict_cost[id] -= label * log(f) + (1 - label) * log(1 - f);

					predict_grad = (label - f) * learn_rate;
					if (d == 0) { // target word has two gradient source, count and predict.
						for (c = 0; c < vector_size; c++) neu1e[c] += predict_grad * syn1neg[c + l2];
						for (c = 0; c < vector_size; c++) neu1e_output[c] += predict_grad * syn0[c + l1];
					} else {
						if (!adagrad) {
							for (c = 0; c < vector_size; c++) neu1e[c] += predict_grad * syn1neg[c + l2];
							for (c = 0; c < vector_size; c++) syn1neg[c + l2] += predict_grad * syn0[c + l1];
						} else {
							for (c = 0; c < vector_size; c++) neu1e[c] += predict_grad * syn1neg[c + l2];
							for (c = 0; c < vector_size; c++) {
								temp = predict_grad * syn0[c + l1];
								syn1neg[c + l2] += temp / sqrt(syn1neg_gradsq[c + l2]);
								syn1neg_gradsq[c + l2] += temp * temp;
							}
						}
					}
				}
			}
			// Compute count error
			l2 = word * vector_size;
			if (hs) {
				count_grad = 0;
				for (c = 0; c < vector_size; c++) count_grad += syn0[c + l1] * syn1neg[c + l2]; //this line is replicated, reduce it may improve speed
			}
			biword = ConcatenateWordIdx(word, last_word);
			biword_count_val = GetCount(biword);

			if (biword_count_val != 0.) {
				count_grad -= log(biword_count_val);
				f_count_grad = (biword_count_val > x_max) ? count_grad : pow(biword_count_val / x_max, alpha) * count_grad;

				count_cost[id] += 0.5 * f_count_grad * count_grad;

				f_count_grad *= learn_rate;
				for (c = 0; c < vector_size; c++) {
					neu1e[c] -= f_count_grad * syn1neg[c  +l2];
					neu1e_output[c] -= f_count_grad * syn0[c + l1];
				}

				if (l1 == l2) {
					for (c = 0; c < vector_size; c++) neu1e[c] += neu1e_output[c];
				} else {
					if (!adagrad) {
						for (c = 0; c < vector_size; c++) syn1neg[c + l2] += neu1e_output[c];
					} else {
						for (c = 0; c < vector_size; c++) {
							syn1neg[c + l2] += neu1e_output[c] / sqrt(syn1neg_gradsq[c + l2]);
							syn1neg_gradsq[c + l2] += neu1e_output[c] * neu1e_output[c];
						}
					}
				}
			} else {
				fprintf(stderr, "%s bigword count is 0.\n", biword);
			}

			if (!adagrad) {
				for (c = 0; c < vector_size; c++) syn0[c + l1] += neu1e[c];
			}
			else {
				for (c = 0; c < vector_size; c++) {
					syn0[c + l1] += neu1e[c] / sqrt(syn0_gradsq[c + l1]);
					syn0_gradsq[c + l1] += neu1e[c] * neu1e[c];
				}
			}

			free(biword);
			pair_count++;
		}
		sentence_position++;
		if (sentence_position >= sentence_length) {
			sentence_length = 0;
			continue;
		}
	}
	fclose(fin);
	free(neu1e);
	pthread_exit(NULL);
}

void SaveParameters() {
	long a, b;
	FILE *fout;
	fout = fopen(output_file, "wb");
	fprintf (fout, "%lld %lld\n", vocab_size, vector_size);
	for (a = 0; a < vocab_size; a++) {
		fprintf(fout, "%s ", vocab[a].word);
		if (binary) for (b = 0; b < vector_size; b++) fwrite(&syn0[a * vector_size + b], sizeof(real), 1, fout);
		else for (b = 0; b < vector_size; b++) fprintf(fout, "%lf ", syn0[a * vector_size + b]);
		fprintf(fout, "\n");
	}
	free(fout);
}

void TrainModel() {
	long long a;
	FILE *fin;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));

	fprintf(stderr, "Strating training using file %s.\n", train_file);
	starting_learn_rate = learn_rate;
	if (vocab_file[0] != 0) ReadVocab(); else {fprintf(stderr, "Vocab file is not speicied.\n"); exit(1);}
	if (output_file[0] == 0) {fprintf(stderr, "Output file is not specified.\n"); exit(1);}
	ConstructBigramTable();
	InitNet();
	if (negative > 0) InitUnigramTable();

	fin = fopen(train_file, "rb");
	if (fin == NULL) {fprintf("Cannot open train file %s.\n", train_file); exit(1);}
	fseek(fin, 0, SEEK_END);
	train_file_size = ftell(fin);
	fclose(fin);

	predict_cost = malloc(sizeof(real) * num_threads);
	count_cost = malloc(sizeof(real) * num_threads);

	start = clock();
	for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
	for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

	SaveParameters();
}

int main()
{
	int i;

	strcpy(vocab_file, "vocab.txt");
	strcpy(train_file, "text8");
	strcpy(cooccur_file, "cooccurrence.shuf.bin");
	strcpy(output_file, "combine2.neg.noada.iter5.bin");

	//Initialize the vocab with vocab_max_size
	vocab = (VWORD *)calloc(vocab_max_size, sizeof(VWORD));

	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
	for (i = 0; i < EXP_TABLE_SIZE; i++) {
		expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
	}

	fprintf(stderr, "Read %lld lines.\n", num_lines);

	TrainModel();
	return 0;
}
