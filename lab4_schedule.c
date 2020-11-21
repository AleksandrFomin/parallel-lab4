#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <sys/time.h>
#include <math.h>

// #define SCHEDULER_TYPE static
// #define SCHEDULER_TYPE dynamic
#define SCHEDULER_TYPE guided
// #define CHUNK_SIZE 1
// #define CHUNK_SIZE 1000
// #define CHUNK_SIZE 50000
// #define CHUNK_SIZE 100000
// #define CHUNK_SIZE 250000
// #define CHUNK_SIZE 450000
#define CHUNK_SIZE 570000

void fill_array(double *arr, int size, double left, double right, unsigned int *seedp)
{
	int i;
	for (i = 0; i < size; i++) {
		arr[i] = rand_r(seedp) / (double)RAND_MAX * (right - left) + left;
	}
}

void print_array(double *arr, int size)
{
	int i;
	printf("arr=[");
	for (i = 0; i < size; i++) {
		printf(" %f", arr[i]);
	}
	printf("]\n");
}

void map_m1(double *arr, int size)
{
	int i;
	#pragma omp parallel for default(none) private(i) shared(arr, size) schedule(SCHEDULER_TYPE, CHUNK_SIZE)
	for (i = 0; i < size; i++) {
		arr[i] = tanh(arr[i]) - 1;
	}
}

void map_m2(double *arr, int size, double *arr_copy)
{
	int i;
	#pragma omp parallel for default(none) private(i) shared(arr, arr_copy, size) schedule(SCHEDULER_TYPE, CHUNK_SIZE)
	for (i = 0; i < size; i++) {
		double prev = 0;
		if (i > 0)
			prev = arr_copy[i - 1];
		arr[i] = sqrt(exp(1.0) * (arr_copy[i] + prev));
	}
}

void copy_arr(double *src, int len, double *dst)
{
	int i;
	#pragma omp parallel for default(none) private(i) shared(src, dst, len) schedule(SCHEDULER_TYPE, CHUNK_SIZE)
	for (i = 0; i < len; i++)
		dst[i] = src[i];
}

void apply_merge_func(double *m1, double *m2, int m2_len)
{
	int i;
	#pragma omp parallel for default(none) private(i) shared(m1, m2, m2_len) schedule(SCHEDULER_TYPE, CHUNK_SIZE)
	for (i = 0; i < m2_len; i++) {
		m2[i] = fabs(m1[i] - m2[i]);
	}
}

// Функция "просеивания" через кучу - формирование кучи
void siftDown(double *numbers, int root, int bottom)
{
	int maxChild; // индекс максимального потомка
	int done = 0; // флаг того, что куча сформирована

	// Пока не дошли до последнего ряда
	while ((root * 2 <= bottom) && (!done)) {
		if (root * 2 == bottom)    // если мы в последнем ряду,
			maxChild = root * 2;    // запоминаем левый потомок
		// иначе запоминаем больший потомок из двух
		else if (numbers[root * 2] > numbers[root * 2 + 1])
			maxChild = root * 2;
		else
			maxChild = root * 2 + 1;
		// если элемент вершины меньше максимального потомка
		if (numbers[root] < numbers[maxChild]) {
			double temp = numbers[root]; // меняем их местами
			numbers[root] = numbers[maxChild];
			numbers[maxChild] = temp;
			root = maxChild;
		} else 
			done = 1; // пирамида сформирована
	}
}

// Функция сортировки на куче
void heapSort(double *numbers, int array_size)
{
	// Формируем нижний ряд пирамиды
	for (int i = (array_size / 2) - 1; i >= 0; i--)
		siftDown(numbers, i, array_size - 1);
	// Просеиваем через пирамиду остальные элементы
	for (int i = array_size - 1; i >= 1; i--) {
		double temp = numbers[0];
		numbers[0] = numbers[i];
		numbers[i] = temp;
		siftDown(numbers, 0, i - 1);
	}
}

double min_not_null(double *arr, int len)
{
	int i;
	double min_val = DBL_MAX;
	for (i = 0; i < len; i++) {
		if (arr[i] < min_val && arr[i] > 0)
			min_val = arr[i];
	}
	return min_val;
}

double reduce(double *arr, int len)
{
	int i;
	double min_val = min_not_null(arr, len);
	double x = 0;
	#pragma omp parallel for default(none) private(i) shared(arr, len, min_val) reduction(+:x) schedule(SCHEDULER_TYPE, CHUNK_SIZE)
	for (i = 0; i < len; i++) {
		if ((int)(arr[i] / min_val) % 2 == 0) {
			double sin_val = sin(arr[i]);
			x += sin_val;
		}
	}
	return x;
}

int main(int argc, char* argv[])
{
	int i, N;
	struct timeval T1, T2;
	long delta_ms;
	double *M1, *M2, *M2_copy;
	int A = 540;
	unsigned int seed1, seed2;
	// double X;

	N = atoi(argv[1]); /* N равен первому параметру командной строки */
	gettimeofday(&T1, NULL); /* запомнить текущее время T1 */

	M1 = malloc(sizeof(double) * N);
	M2 = malloc(sizeof(double) * N / 2);
	M2_copy = malloc(sizeof(double) * N / 2);

	for (i = 0; i < 50; i++) /* 50 экспериментов */
	{
		seed1 = i;
		seed2 = i;
		fill_array(M1, N, 1, A, &seed1);
		fill_array(M2, N / 2, A, 10 * A, &seed2);

		// printf("Fill arrays\n");
		// print_array(M1, N);
		// print_array(M2, N / 2);
		
		map_m1(M1, N);
		copy_arr(M2, N / 2, M2_copy);
		map_m2(M2, N / 2, M2_copy);

		// printf("Map\n");
		// print_array(M1, N);
		// print_array(M2, N / 2);

		apply_merge_func(M1, M2, N / 2);
		// printf("Merge\n");
		// print_array(M2, N / 2);

		heapSort(M2, N / 2);
		// printf("Sort\n");
		// print_array(M2, N / 2);

		reduce(M2, N / 2);
		// printf("X = %f\n", X);
	}
	gettimeofday(&T2, NULL); /* запомнить текущее время T2 */

	delta_ms = 1000 * (T2.tv_sec - T1.tv_sec) + (T2.tv_usec - T1.tv_usec) / 1000;
	printf("%d %ld\n", N, delta_ms); /* T2 - T1 */

	free(M1);
	free(M2);
	free(M2_copy);

	return 0;
}