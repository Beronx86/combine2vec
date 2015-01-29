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
#define MAX_CODE_LENGTH 40
#define MAX_EXP 6
#define EXP_TABLE_SIZE 1000

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

int enable_predict = 1;
int enable_count = 1;
int verbose = 2; // 0, 1, or 2
int binary = 1;
int num_threads = 1; // pthreads
int adagrad = 0;
int num_iter = 25; // Number of full passes through cooccurrence matrix
int save_gradsq = 0; // By default don't save squared gradient values
int vector_size = 100; // Word vector size
long long num_lines = 0, *lines_per_thread, vocab_size = 0, vocab_max_size = 2500;
long long word_count_actual = 0;
char vocab_file[MAX_STRING_LENGTH], train_file[MAX_STRING_LENGTH], output_file[MAX_STRING_LENGTH];
real learn_rate = 0.025, starting_learn_rate; // Initial learning rate
real alpha = 0.75, x_max = 100.0; // Weighting function parameters, not extremely sensitive to corpus, though may need adjustment for very small or very large corpora
real *syn0, *syn1, *syn1neg, *gradsq, *expTable; //syn0 input word embeding (the i in glove Xij)
real *cost;
VWORD *vocab;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

/* Efficient string comparison */
int scmp( char *s1, char *s2 ) {
    while(*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return(*s1 - *s2);
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
		for (a = 0; a < vocab_size; a++) for (b = 0; b < vector_size; b++)
			syn1[a * vector_size + b] = 0;
	}
	if (negative > 0){
		a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * vector_size * sizeof(real));
		if (syn1neg == NULL) {fprintf(stderr, "Error allocating memory for syn1\n"), exit(1);}
		for (a = 0; a < vocab_size; a++) for (b = 0; b < vector_size; b++)
			syn1neg[a * vector_size + b] = 0;
	}
	for (a = 0; a < vocab_size; a++) for (b = 0; b < vector_size; b++) {
		next_random = next_random * (unsigned long long)25214903917 + 11;
		syn0[a * vector_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / vector_size;
	}
	CreateBinaryTree();
}


void *TrainModelThread(void *vid) {
	long long a, b, c, d;
	long long l1, l2, word1, word2, target, label;
	long long local_iter = num_iter, word_count = 0, last_word_count = 0;
	long long id = (long long) vid;
	unsigned long long next_random = (long long) id;
	real f, predict_grad, count_grad, f_count_grad;
	real temp1, temp2;
	real *neu1e = (real *)calloc(vector_size, sizeof(real));
	CREC cr;
	FILE *fin;
	clock_t now;
	fin = fopen(train_file, "rb");
	fseeko(fin, (num_lines / num_threads * id) * sizeof(CREC), SEEK_SET);
	cost[id] = 0;

	// for (a = 0; a < lines_per_thread[id]; a++) {
	while(1){
		if (word_count - last_word_count > 10000) {
			word_count_actual += word_count - last_word_count;
			last_word_count = word_count;
			if (verbose > 1) {
				now = clock();
				if (!adagrad)
				{
					fprintf(stderr, "%cLearning Rate: %f Progess: %.2f%% Words/thread/sec: %.2fk ", 13,
							learn_rate, word_count_actual / (real)(num_iter * num_lines + 1) * 100,
							word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
					fflush(stderr);
				} else {
					fprintf(stderr, "%cProgess: %.2f%% Words/thread/sec: %.2fk ", 13,
							word_count_actual / (real)(num_iter * num_lines + 1) * 100,
							word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
					fflush(stderr);
				}
			}
			if (!adagrad) {
				learn_rate = starting_learn_rate * (1 - word_count_actual / (real)(num_iter * num_lines + 1));
				if (learn_rate < starting_learn_rate * 0.0001) learn_rate = starting_learn_rate * 0.0001;
			}
		}

		if (feof(fin) || (word_count >= lines_per_thread[id])) {
			word_count_actual += word_count - last_word_count;
			local_iter--;
			if (local_iter == 0) break;
			word_count = 0;
			last_word_count = 0;
			fseeko(fin, (num_lines / num_threads * id) * sizeof(CREC), SEEK_SET);
		}

		fread(&cr, sizeof(CREC), 1, fin);
		if (feof(fin)) continue;
		word_count++;

		word1 = cr.word1 - 1LL; // input word (current word)
		l1 = word1 * vector_size;
		word2 = cr.word2 - 1LL; // output word (context word)

		// Optimize predict error
		if (enable_predict) {
			for (c = 0; c < vector_size; c++) neu1e[c] = 0;

			if (hs) for (d = 0; d < vocab[word2].codelen; d++) {
				f = 0;
				l2 = vocab[word2].point[d] * vector_size;

				for (c = 0; c < vector_size; c++) f += syn0[c + l1] * syn1[c + l2];
				if (f <= -MAX_EXP) continue;
				else if (f >= MAX_EXP) continue;
				else f = expTable[(int)(f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2)];

				predict_grad = (1 - vocab[word2].code[d] - f) * learn_rate;
				for (c = 0; c < vector_size; c++) neu1e[c] += predict_grad * syn1[c + l2];
				for (c = 0; c < vector_size; c++) syn1[c + l2] += predict_grad * syn0[c + l1];
			}

			if (negative > 0) for (d = 0; d < negative + 1; d++) {
				if (d == 0) {
					target = word2;
					label = 1;
				} else {
					next_random = next_random * (unsigned long long)25214903917 + 11;
					target = table[(next_random >> 16) % table_size];
					if (target == 0) target = next_random % (vocab_size - 1) + 1;
					if (target == word2) continue;
					label = 0;
				}

				l2 = target * vector_size;
				f = 0;
				for (c = 0; c < vector_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
				if (f > MAX_EXP) predict_grad = (label - 1) * learn_rate;
				else if (f < -MAX_EXP) predict_grad = (label - 0) * learn_rate;
				else predict_grad = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * learn_rate;
				for (c = 0; c < vector_size; c++) neu1e[c] += predict_grad * syn1neg[c + l2];
				for (c = 0; c < vector_size; c++) syn1neg[c + l2] += predict_grad * syn0[c + l1];
			}
			for (c = 0; c < vector_size; c++) syn0[c + l1] += neu1e[c];
		}

		// Optimize count error
		if (enable_predict) {
			l2 = word2 * vector_size; //no bias version
			count_grad = 0;
			for (c = 0; c < vector_size; c++) count_grad += syn0[b + l1] * syn0[b + l2]; // current and context word are represented by the same vector space
			count_grad -= log(cr.val); //no bias version
			f_count_grad = (cr.val > x_max) ? count_grad : pow(cr.val / x_max, alpha) * count_grad;
			cost[id] += 0.5 * f_count_grad * count_grad;

			f_count_grad *= learn_rate;
			for (c = 0; c < vector_size; c++) {
				temp1 = f_count_grad * syn0[c + l2];
				temp2 = f_count_grad * syn0[c + l1];
				if (!adagrad) {
					syn0[b + l1] -= temp1;
					syn0[b + l2] -= temp2;
				}
			}
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
	long long file_size;
	long long a, b, c, d;
	FILE *fin, *fo;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));

	fprintf(stderr, "Strating training using file %s.\n", train_file);
	starting_learn_rate = learn_rate;
	if (vocab_file[0] != 0) ReadVocab(); else {fprintf(stderr, "Vocab file is not speicied.\n"); exit(1);}
	if (output_file[0] == 0) {fprintf(stderr, "Output file is not specified.\n"); exit(1);}
	InitNet();
	if (negative > 0) InitUnigramTable();

	fin = fopen(train_file, "rb");
	if (fin == NULL) {fprintf(stderr, "Unable to open coocurrence file %s.\n", train_file); exit(1);}
	fseek(fin, 0, SEEK_END);
	file_size = ftello(fin);
	num_lines = file_size / sizeof(CREC);
	fclose(fin);
	fprintf(stderr, "Read %lld lines.\n", num_lines);

	lines_per_thread = (long long *) malloc (num_threads * sizeof(long long));
	for (a = 0; a < num_threads - 1; a++) lines_per_thread[a] = num_lines / num_threads;
	lines_per_thread[a] = num_lines / num_threads + num_lines % num_threads;
	cost = malloc(sizeof(real) * num_threads);

	start = clock();
	for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
	for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

	SaveParameters();
}

int main()
{
	int i;

	strcpy(vocab_file, "vocab.txt");
	strcpy(train_file, "cooccurrence.shuf.bin");
	strcpy(output_file, "combine.out.bin");

	//Initialize the vocab with vocab_max_size
	vocab = (VWORD *)calloc(vocab_max_size, sizeof(VWORD));

	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
	for (i = 0; i < EXP_TABLE_SIZE; i++) {
		expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
	}

	TrainModel();
	return 0;
}
